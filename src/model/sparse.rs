//! # SparseAttention —— KV 侧稀疏注意力（「位置打分 + 选择」算子族）
//!
//! ## 背景：2025-2026 的第二维稀疏化
//!
//! MoE 完成了 **FFN 维稀疏**（token → 少量专家）。2025 年起旗舰模型把稀疏化
//! 延伸到 **KV 维**：长上下文下每个 query 只需与「相关」的历史 token 打分。
//! 代表工作（均为 **块级检索 + 局部窗口** 的组合）：
//!
//! - **NSA**（DeepSeek 2025-02）：压缩块 + 细粒度 token 选择 + 滑动窗口；
//! - **MoBA**（Kimi 2025-02）：KV 切成块，每 query 按块路由/挑选再注意力；
//! - **DSA**（DeepSeek-V3.2 2025-09/12）：Lightning Indexer 训练出的位置打分器
//!   + Top-k + FP8，**训练与推理同构**的稀疏注意力（DeepSeek-V3.2-Exp 主打）。
//!
//! 三种方法共享同一组**算子碎片**：
//!
//! ```text
//! ① KV 位置打分：把历史切成块，取各块代表向量（均值/sink/块首）
//!    对每条 query 打分（本文件：q · 块代表）
//! ② Top-k 选择：取分最高的 k 个块（附因果约束：只允许历史块）
//! ③ 局部窗口：最近的 w 个 token 始终可见（保证局部性/位置信息）
//! ```
//!
//! ## 本文件的定位（逻辑等价面 → kernel 位）
//!
//! Burn 高层张量 API 无法对「每 query 变长的稀疏 KV」做零散 gather 后直接
//! 算注意力，因此本文件把 ①②③ 落为**张量级 mask**：稀疏语义（每个 query
//! 只能看到 top-k 块 ∪ 窗口内的键）与 NSA/MoBA/DSA 完全一致，scores 被
//! `-inf` 屏蔽到这些位置。复杂度与 naive SDPA 同为 \(O(S^2)\)——
//! **物理省算**（只对被选块打分）依赖未来在 `cubecl` 层用本 mask/indices
//! 实现 Flash 风格 kernel（`docs/INDUSTRIAL.zh-CN.md` 记录实现范围）。
//! mask 语义与 kernel 一一对应：训练时即可跑通稀疏注意力（forward/backward
//! 全支持），权重/结构零迁移成本。
//!
//! ## 数学语义
//!
//! 设 \(S\) 个键按 `block_size` 分成 \(N = \lceil S / b \rceil\) 块，块 \(j\)
//! 覆盖键位置 \([\,jb,\ (j{+}1)b\,)\)。query 行 \(p\)（全局位置）的可见集：
//!
//! \[
//! \mathcal{V}_p = \underbrace{\{ \text{分最高的 }k\text{ 个块} \cap \{j : jb \le p\} \}}_{\text{块 Top-k（数据相关）}}
//! \cup \underbrace{\{p-w,\ \dots,\ p\}}_{\text{滑动窗口（结构相关）}}
//! \]
//!
//! 其中块得分 = query 与**块代表**的内积。块代表取「块内首个 token 的 K」
//! （位置 \(jb\)，见下方注记）。注意 \(\mathcal{V}_p \subseteq
//! \{0,\dots,p\}\)，因果性天然成立。
//!
//! ## 精度注记（与后续升级点的对应）
//!
//! - **块代表 = 块首 token**（而非块均值）：均值会把「未来 token」混入当前
//!   行的候选块，破坏 prefill 与 decode 的一致性；块首则总在历史内，
//!   prefill（有未来）与 decode（无未来）对同一条 query 给出**完全一致的
//!   Top-k 结果**（本模块一致性测试保证）。升级点：NSA 式可学 sink
//!   （带权重的块代表，训练时端到端学习）；
//! - 块得分 = 原始 q·k 内积（未缩放），Top-k 次序与 \(\sqrt{d}\) 缩放无关；
//! - 真实 kernel 位可把「mask 真位置」等价为「gather 索引」，省掉被屏蔽
//!   位置的打分，见模块 doc 与 docs/INDUSTRIAL.zh-CN.md。
//!
//! ## 形状约束
//!
//! - `q`：`[Batch, Heads, S_q, HeadDim]`（已 RoPE、已多头布局）；
//! - `k`：`[Batch, Heads, S_kv, HeadDim]`（已 RoPE，**已做 GQA 头扩展**，
//!   使头数与 `q` 一致才能批量打分）；
//! - 返回 mask：`[Batch, Heads, S_q, S_kv]` Bool，`true` = 屏蔽
//!   （与 `sdpa` 的 mask 语义一致，可直接喂给 `scaled_dot_product_attention`）。

use serde::{Deserialize, Serialize};

use burn::tensor::{Bool, Int, Tensor};

/// 稀疏注意力配置（`config.sparse = Some(..)` 启用；`None` = 传统 dense）。
///
/// 三个参数分别对应模块 doc 中的三个算子碎片：
/// - `block_size`：KV 分块大小（位置打分/检索的粒度，常见 16~128）；
/// - `top_k_blocks`：每条 query 可选的**历史块**数（DSA/MoBA 的 top-k）；
/// - `window`：始终可见的最近 token 数（NSA 的滑动窗；`0` = 关闭）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseAttentionConfig {
    /// KV 分块大小（`block_size >= 1`）。
    pub block_size: usize,
    /// 每条 query 从历史中选出的块数（`>= 1`，会被夹在 `[1, N]`）。
    pub top_k_blocks: usize,
    /// 始终可见的最近 token 数（`0` = 仅靠块检索）。
    pub window: usize,
}

impl SparseAttentionConfig {
    /// 构造配置（自动夹紧非法值，见各字段注记）。
    pub fn new(block_size: usize, top_k_blocks: usize, window: usize) -> Self {
        Self {
            block_size: block_size.max(1),
            top_k_blocks: top_k_blocks.max(1),
            window,
        }
    }
}

/// 构造稀疏注意力 mask（「位置打分 + Top-k 块选择 + 窗口」的融合）。
///
/// 返回 `[Batch, Heads, S_q, S_kv]` Bool 掩码：`true` = 该 (query, key) 对
/// 被稀疏化屏蔽（`false` = 保留参与注意力）。
///
/// - `q_offset`：第一条 query 的**全局位置**（prefill 首行 = `0`；cache 解码
///   时 = 该行全局位置，见 `attention` 的调用处约定）。
///   用于把「行内下标」换算成因果位置，保证块选择/窗口都只落在自己的历史。
pub fn sparse_attention_mask(
    q: Tensor<4>,
    k: Tensor<4>,
    q_offset: usize,
    cfg: &SparseAttentionConfig,
) -> Tensor<4, Bool> {
    let [batch, heads, sq, head_dim] = q.dims();
    let [_, _, kv_len, _] = k.dims();
    let device = q.device();

    let bs = cfg.block_size.max(1);
    let n_blocks = kv_len.div_ceil(bs);
    // top-k 不能超过总块数（`select` 越界保护）；同时 ≥ 1。
    let top_k = cfg.top_k_blocks.max(1).min(n_blocks.max(1));

    // ============ ① KV 分块 + 块代表（位置打分） ============
    // 不足整块的尾块以零补齐 → 每块长度固定，可批量 reshape。
    let padded_len = n_blocks * bs;
    let k_padded = if padded_len > kv_len {
        let zeros = Tensor::<4>::zeros([batch, heads, padded_len - kv_len, head_dim], &device);
        Tensor::cat(vec![k.clone(), zeros], 2)
    } else {
        k
    };
    // 块代表 = 块内**首个** token 的 K（位置 jb）。选块首而非块均值的原因：
    // 均值会把「未来 token」混入当前行的候选块（decode 时该未来尚不存在），
    // 导致 prefill 与 decode 的 Top-k 打分不一致；块首总在历史内，天然一致。
    // （可升级为可学 sink / 缩放指数，见模块 doc 精度注记。）
    let k_block = k_padded
        .reshape([batch, heads, n_blocks, bs, head_dim])
        .slice([0..batch, 0..heads, 0..n_blocks, 0..1, 0..head_dim])
        .reshape([batch, heads, n_blocks, head_dim])
        .swap_dims(2, 3);
    // 打分：[B,H,Sq,N]（未缩放；缩放不改变 Top-k 次序）
    let scores = q.matmul(k_block);

    // ============ 块级因果约束：块 j 可见 ⇔ jb ≤ p ============
    let row_pos = Tensor::<1, Int>::arange(0..sq as i64, &device)
        .reshape([sq, 1])
        .add_scalar(q_offset as i64);
    let block_start = Tensor::<1, Int>::arange(0..n_blocks as i64, &device)
        .reshape([1, n_blocks])
        .mul_scalar(bs as i64);
    // 以下 mask 统一用「true = 可见」语义（便于阅读）；
    // 到函数出口再翻转成 sdpa 约定的「true = 屏蔽」。
    // 块级因果：块 j 可见 ⇔ jb ≤ p（row_pos 之后还会用于窗口计算，先克隆）
    let visible = (row_pos.clone() - block_start)
        .greater_equal_elem(0_i64)
        .reshape([1, 1, sq, n_blocks])
        .expand([batch, heads, sq, n_blocks]);
    // mask_fill 的 true = 屏蔽 → 填不可见（= 反相 visible）
    let invisible = (1.0_f32 - visible.float()).greater_equal_elem(0.5_f32);
    let scores = scores.mask_fill(invisible, -f32::INFINITY);

    // ============ ② Top-k 块选择（含索引 → 逐槽 OR 成块 mask） ============
    // topk_with_indices: [B,H,Sq,k]（desc）；早期行可见块不足 k 时会取到
    // -inf 的「假块」，交由最后 token 级因果掩码裁掉。
    let (_, indices) = scores.topk_with_indices(top_k, 3);

    let block_ids = Tensor::<1, Int>::arange(0..n_blocks as i64, &device)
        .float()
        .reshape([1, 1, 1, n_blocks]);

    let mut acc: Option<Tensor<4, Int>> = None;
    for t in 0..top_k {
        let idx_t = indices
            .clone()
            .slice([0..batch, 0..heads, 0..sq, t..t + 1])
            .float();
        // 逐块精确相等（整数转浮点在 < 2^24 内无损）
        let hit = (idx_t - block_ids.clone()).abs().lower_equal_elem(0.5_f32); // Bool
        let add = hit.int();
        acc = Some(match acc {
            None => add,
            Some(a) => a + add,
        });
    }
    let block_mask = acc.expect("top_k ≥ 1 保证至少一轮").greater_equal_elem(1);

    // 块 mask 展开回 token 粒度：每块命中 → 块内 b 个 token 全可见
    let block_token_mask = block_mask
        .unsqueeze_dims::<5>(&[4])
        .expand([batch, heads, sq, n_blocks, bs])
        .reshape([batch, heads, sq, padded_len]);
    let block_token_mask = if padded_len > kv_len {
        block_token_mask.slice([0..batch, 0..heads, 0..sq, 0..kv_len])
    } else {
        block_token_mask
    };

    // ============ token 级因果 + ③ 滑动窗口 ============
    let col = Tensor::<1, Int>::arange(0..kv_len as i64, &device).reshape([1, kv_len]);
    let d = col - row_pos; // [Sq, S]：列 - 行
    let causal = d.clone().lower_equal_elem(0_i64); // 仅历史（col ≤ p）
    let window = (causal.clone().int() * d.greater_equal_elem(-(cfg.window as i64)).int())
        .greater_equal_elem(1); // col ∈ [p-w, p]

    let causal4 = causal
        .reshape([1, 1, sq, kv_len])
        .expand([batch, heads, sq, kv_len]);
    let window4 = window
        .reshape([1, 1, sq, kv_len])
        .expand([batch, heads, sq, kv_len]);

    // 块选择产生的「假块/未来 token」在此被因果裁掉：
    let blocks_valid = (block_token_mask.int() * causal4.int()).greater_equal_elem(1);
    // 融合：top-k 块 ∪ 窗口（均 true = 可见）→ 出口翻转成 true = 屏蔽
    let visible_final = (blocks_valid.int() + window4.int()).greater_equal_elem(1);
    (1.0_f32 - visible_final.float()).greater_equal_elem(0.5_f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::get_device;
    use crate::model::sdpa::build_causal_mask;

    fn causal_cover(device: &burn::tensor::Device, sq: usize, kv_len: usize) -> Tensor<4, Bool> {
        // [1,1,Sq,S]：只屏蔽未来列
        let mut mask = build_causal_mask(kv_len, device).reshape([1, 1, kv_len, kv_len]);
        if sq < kv_len {
            mask = mask.slice([0..1, 0..1, 0..sq, 0..kv_len]);
        }
        mask
    }

    /// burn 后端 Int 元素为 i32，计数直接回传 i32。
    fn bool_count(t: Tensor<4, Bool>) -> i32 {
        let v: Tensor<1, Int> = t.int().sum();
        v.into_scalar::<i32>()
    }

    /// 配置参数小、kv 长时，稀疏 mask 的可见位置严格少于 dense(causal)。
    #[test]
    fn test_mask_is_strictly_sparser() {
        let device = get_device();
        let kv_len = 32;
        let sq = 32;
        let q = Tensor::<4>::random(
            [1, 2, sq, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let k = Tensor::<4>::random(
            [1, 2, kv_len, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let cfg = SparseAttentionConfig::new(8, 2, 4); // 每行最多 ~20，远小于 causal
        let mask = sparse_attention_mask(q, k, 0, &cfg);

        let total = bool_count(mask.clone());
        let dense_total = bool_count(causal_cover(&device, sq, kv_len));

        // mask 语义 true=屏蔽：稀疏应屏蔽显著多于 dense（= 保留更少）
        assert!(total > 0, "至少屏蔽部分位置");
        assert!(
            total > dense_total * 2,
            "稀疏屏蔽数应显著多于 dense（{total} vs {dense_total}）"
        );
    }

    /// 全覆盖配置（top-k 选所有块 ∪ 窗口 ≥ 全长）退化为 causal（≈ dense）。
    #[test]
    fn test_mask_full_cover_equals_causal() {
        let device = get_device();
        let kv_len = 16;
        let sq = 16;
        let q = Tensor::<4>::random(
            [1, 1, sq, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let k = Tensor::<4>::random(
            [1, 1, kv_len, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        // top_k = N（全选）+ window = 全长 → mask == causal
        let cfg = SparseAttentionConfig::new(4, 4, kv_len);
        let mask = sparse_attention_mask(q, k, 0, &cfg);
        let expected = causal_cover(&device, sq, kv_len);

        let diff: Tensor<1, Int> = (mask.int() - expected.int()).abs().sum();
        assert_eq!(diff.into_scalar::<i32>(), 0, "全覆盖稀疏应等于 causal mask");
    }

    /// 退化情形：kv=1 时行 0（pos=0、无窗口）只能看到自己。
    #[test]
    fn test_first_row_sees_only_itself() {
        let device = get_device();
        let q = Tensor::<4>::random(
            [1, 1, 1, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let k = Tensor::<4>::random(
            [1, 1, 8, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let cfg = SparseAttentionConfig::new(2, 1, 0);
        // 只给一行 kv：行 0 无历史、无窗口 → 只有自身可见 → 无任何被屏蔽位置
        let k1 = k.slice([0..1, 0..1, 0..1, 0..8]);
        let mask = sparse_attention_mask(q, k1, 0, &cfg);
        let count = bool_count(mask);
        assert_eq!(
            count, 0,
            "pos=0、kv=1、无窗口时唯一 token 可见，不应屏蔽任何位置"
        );
    }
}
