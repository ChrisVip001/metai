//! # GroupedQueryAttention —— 分组查询注意力组合算子（GQA）
//!
//! 自注意力是 Transformer 捕捉「token 与 token 之间关系」的核心机制。
//! 对查询序列的每个 token：
//!
//! \[
//! \text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V
//! \]
//!
//! 本结构是「**组合算子**」：它本身不重复实现数学细节，而是按固定顺序
//! 编排一组可独立替换/升级的子算子（详见下文流水线）。
//!
//! ## GQA 与 KV Cache 的关系
//!
//! - **MHA**：每头独立 K/V → 显存与计算都随头数线性增长；
//! - **MQA**：全部头共享 1 组 K/V → KV Cache 最小，但表达力偏弱；
//! - **GQA（本实现）**：\(n\_heads\) 个查询头分成 \(n\_kv\_heads\) 组，
//!   每组共享一组 K/V。推理时 KV Cache 只按 KV 头数量存储，比 MHA 省
//!   \(n\_heads/n\_kv\_heads\) 倍显存；效果接近 MHA。
//!
//! 权重布局：`w_q: [n_heads·head_dim, hidden]`，
//! `w_k / w_v: [n_kv_heads·head_dim, hidden]`，`w_o: [hidden, n_heads·head_dim]`。
//!
//! ## 前向算子流水线（各步对应独立算子文件）
//!
//! ```text
//! x [B, S, H]
//!  │
//!  ├─ [算子] QKV 投影（本结构内的 w_q / w_k / w_v，见下）
//!  ├─ [算子] HeadOps::split_heads         → [B, H, S, d_k]   （heads.rs）
//!  ├─ [算子] RoPE::forward_with_offset    → 旋转 Q / K       （positional_encoding.rs）
//!  │
//!  ├─ 有 Cache：KV Cache 更新 → HeadOps::repeat_kv（GQA 头扩展）→ SDPA(无 mask)
//!  └─ 无 Cache：HeadOps::repeat_kv → mask 组装 → SDPA（因果 + padding 掩码）
//!  │
//!  ├─ [算子] HeadOps::merge_heads         → [B, S, H]        （heads.rs）
//!  └─ [算子] 输出投影 w_o
//! ```
//!
//! QKV 投影本身可进一步升级为**融合矩阵**（把 `w_q/w_k/w_v` 拼接成一个
//! 大 Linear，一次 GEMM 出三个矩阵，即流行的 QKV-fused kernel）；
//! 该改动只影响本结构内部，不会波及 `TransformerBlock` 与 checkpoint
//! 之外的部分。
//!
//! ## Mask 语义说明
//!
//! - **无 Cache（训练 / Prefill）**：需要因果 mask（自回归：当前位置不能看
//!   未来）与 padding mask（pad token 不参与 attention）；
//! - **有 Cache（解码）**：K/V 只来自 Cache（历史 token），Q 只与历史打分，
//!   天然满足因果性，**无需再构造 mask**；同时 GQA 的 K/V 头扩展发生在
//!   Cache 读出之后（每次只对最新 token 做一次 repeat）。

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Bool, Tensor};

use crate::model::heads;
use crate::model::mla::MultiLatentAttention;
use crate::model::positional_encoding::RoPE;
use crate::model::sdpa;
use crate::model::sparse::{sparse_attention_mask, SparseAttentionConfig};

/// 注意力结构变体容器（`TransformerBlock` 的装配点，形态同 `norm::Norm`）。
///
/// - `Gqa`：分组查询注意力（默认，本文件）；
/// - `Mla`：低秩潜空间注意力（DeepSeek-V2 族，见 [`crate::model::mla`]）。
///
/// 替换变体只改构造处（`TransformerBlock::new` 依 `config.attn` 分发），
/// 块级流水线与推理接口不变。
#[derive(Module, Debug)]
pub enum Attention {
    Gqa(GroupedQueryAttention),
    Mla(MultiLatentAttention),
}

impl Attention {
    /// 按结构变体分发到具体实现的编排前向（签名与块内调用一致）。
    pub fn forward(
        &self,
        x: Tensor<3>,
        rope: &RoPE,
        mask: Option<Tensor<2, Bool>>,
        cache: Option<&mut crate::infer::cache::KVCache>,
    ) -> Tensor<3> {
        match self {
            Attention::Gqa(gqa) => gqa.forward(x, rope, mask, cache),
            Attention::Mla(mla) => mla.forward(x, rope, mask, cache),
        }
    }
}

/// GQA 注意力组合算子。
///
/// 结构字段（保持稳定，checkpoint 权重路径依赖这些字段名）：
/// - `w_q`：查询投影 `[n_heads·head_dim, hidden]`；
/// - `w_k` / `w_v`：键/值投影 `[n_kv_heads·head_dim, hidden]`；
/// - `w_o`：输出投影 `[hidden, n_heads·head_dim]`；
/// - `n_heads / n_kv_heads / head_dim`：头数等超参数（非参数）。
#[derive(Module, Debug)]
pub struct GroupedQueryAttention {
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// KV 侧稀疏注意力配置（`None` = dense，见 [`crate::model::sparse`]）。
    /// 非权重、不参与 checkpoint（`Ignored` 容器）。
    #[module(skip)]
    pub sparse: Option<SparseAttentionConfig>,
}

impl GroupedQueryAttention {
    /// 构造 GQA：
    /// `hidden_dim → n_heads·head_dim`（Q）与 `n_kv_heads·head_dim`（K/V），
    /// 输出投影回到 `hidden_dim`。
    pub fn new(
        hidden_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        device: &burn::tensor::Device,
    ) -> Self {
        let head_dim = hidden_dim / n_heads;
        Self {
            w_q: LinearConfig::new(hidden_dim, n_heads * head_dim)
                .with_bias(false)
                .init(device),
            w_k: LinearConfig::new(hidden_dim, n_kv_heads * head_dim)
                .with_bias(false)
                .init(device),
            w_v: LinearConfig::new(hidden_dim, n_kv_heads * head_dim)
                .with_bias(false)
                .init(device),
            w_o: LinearConfig::new(n_heads * head_dim, hidden_dim)
                .with_bias(false)
                .init(device),
            n_heads,
            n_kv_heads,
            head_dim,
            sparse: None,
        }
    }

    /// 设定 KV 侧稀疏注意力（`None` = 恢复 dense；默认 dense）。
    pub fn set_sparse(&mut self, sparse: Option<SparseAttentionConfig>) {
        self.sparse = sparse;
    }

    /// GQA 前向传播（子算子编排）。
    ///
    /// - `x`：`[Batch, Seq, Hidden]`（已由 `attn_norm` 归一化）；
    /// - `rope`：共享的 RoPE 算子（含预计算频率表）；
    /// - `mask`：padding mask `[Batch, Seq]`，`true` = pad（仅无 Cache 时生效）；
    /// - `cache`：可选 KV Cache（推理解码时传入，走增量路径）。
    pub fn forward(
        &self,
        x: Tensor<3>,
        rope: &RoPE,
        mask: Option<Tensor<2, Bool>>,
        cache: Option<&mut crate::infer::cache::KVCache>,
    ) -> Tensor<3> {
        let [batch_size, seq_len, _hidden_dim] = x.dims();

        // ============ 算子 1：QKV 投影（三个 Linear 共享输入） ============
        let q = self.w_q.forward(x.clone());
        let k = self.w_k.forward(x.clone());
        let v = self.w_v.forward(x);

        // ============ 算子 2：头拆分 → [B, H, S, d_k] ============
        let q = heads::split_heads(q, self.n_heads);
        let k = heads::split_heads(k, self.n_kv_heads);
        let v = heads::split_heads(v, self.n_kv_heads);

        // ============ 算子 3：RoPE 旋转（位置由 cache 长度决定） ============
        let offset = cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);
        let q = rope.forward_with_offset(q, offset);
        let k = rope.forward_with_offset(k, offset);

        // GQA：每组 KV 头对应的查询头数（repeat 倍数）
        let n_rep = self.n_heads / self.n_kv_heads;
        // 注意力缩放因子 1/√d_k
        let scale = (self.head_dim as f32).sqrt().recip();

        if let Some(cache) = cache {
            let padding = cache.append_padding(mask, batch_size, seq_len, &q.device());
            // ============ 算子 4：KV Cache 更新（只存 KV 头，不重复） ============
            cache.update(k, v);

            // ============ 算子 6：SDPA ============
            // dense / 多行 mask 位需要先 `get_full`（全量拼接）；单行稀疏解码
            // 由 sparse_exec 直接以 cache 为数据源做「页读取」（对齐时 O(L)，
            // 否则内部回退 get_full），这里不再无条件执行 O(S_kv) 拼接。
            let out = if let Some(sparse_cfg) = self.sparse.as_ref() {
                if batch_size == 1 && seq_len == 1 && padding.is_none() {
                    // 稀疏解码（主路径）：紧凑 gather 物理位（sparse_exec.rs）。
                    // 只对「Top-k 块 ∪ 窗口」的 KV 做注意力，物理省掉
                    // 其余 O(S_kv) 打分/softmax/加权；块与 cache 页对齐时
                    // 走页读取（head_rows 打分 + gather_spans 取 KV）。
                    crate::model::sparse_exec::sparse_decode_attention_cached(
                        q, cache, n_rep, sparse_cfg,
                    )
                } else {
                    // 多行（speculative 验证等）→ mask 位参考实现：
                    // q 批的全局起点 = cache 更新前长度（与 RoPE 的 offset
                    // 同源）；GQA 头扩展后再稀疏打分。
                    let (k_full, v_full) = cache.get_full();
                    let k = heads::repeat_kv(k_full, n_rep);
                    let v = heads::repeat_kv(v_full, n_rep);
                    let sp = sparse_attention_mask(q.clone(), k.clone(), offset, sparse_cfg);
                    let sp = if let Some(p) = padding {
                        sp.bool_or(p)
                    } else {
                        sp
                    };
                    sdpa::scaled_dot_product_attention(q, k, v, scale, Some(sp))
                }
            } else {
                // dense 解码：天然因果，无 mask
                let (k_full, v_full) = cache.get_full();
                let k = heads::repeat_kv(k_full, n_rep);
                let v = heads::repeat_kv(v_full, n_rep);
                {
                    let mask = sdpa::cached_mask(offset, seq_len, &q.device(), padding);
                    sdpa::scaled_dot_product_attention(q, k, v, scale, mask)
                }
            };

            // ============ 算子 7：合并头 + 输出投影 ============
            let out = heads::merge_heads(out);
            self.w_o.forward(out)
        } else {
            // ============ 算子 5：GQA 头扩展 ============
            let k = heads::repeat_kv(k, n_rep);
            let v = heads::repeat_kv(v, n_rep);

            // ============ 算子 6：mask 组装 + SDPA（训练 / Prefill） ============
            let attn_mask: Option<Tensor<4, Bool>> = if let Some(sparse_cfg) = self.sparse.as_ref()
            {
                // 稀疏注意力：位置打分 + Top-k 块 + 滑动窗（mask 内含因果），
                // 再叠加 padding（若存在）。返回的 mask 直接喂 SDPA。
                let sp = sparse_attention_mask(q.clone(), k.clone(), 0, sparse_cfg);
                match mask {
                    Some(pad_mask) => {
                        let pad = pad_mask
                            .unsqueeze::<4>()
                            .reshape([batch_size, 1, 1, seq_len]);
                        Some((sp.int() + pad.int()).greater_equal_elem(1))
                    }
                    None => Some(sp),
                }
            } else if seq_len > 1 {
                // 因果 mask：[1,1,S,S]，true=屏蔽
                let device = q.device();
                let causal =
                    sdpa::build_causal_mask(seq_len, &device).reshape([1, 1, seq_len, seq_len]);

                match mask {
                    // padding mask：[B,S] -> [B,1,1,S]，与因果 mask 合并
                    Some(pad_mask) => {
                        let pad_expanded = pad_mask
                            .unsqueeze::<4>()
                            .reshape([batch_size, 1, 1, seq_len]);
                        Some(sdpa::combine_masks(
                            causal,
                            pad_expanded,
                            batch_size,
                            seq_len,
                        ))
                    }
                    None => Some(causal),
                }
            } else {
                None
            };

            let out = sdpa::scaled_dot_product_attention(q, k, v, scale, attn_mask);

            // ============ 算子 7：合并头 + 输出投影 ============
            let out = heads::merge_heads(out);
            self.w_o.forward(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::get_device; // Use project defined backend
    use crate::infer::cache::KVCache;
    use crate::model::positional_encoding::RoPE;

    #[test]
    fn test_paged_attention_consistency() {
        let device = get_device();
        let hidden_dim = 16;
        let n_heads = 4;
        let n_kv_heads = 2; // GQA
        let head_dim = hidden_dim / n_heads;

        let attn = GroupedQueryAttention::new(hidden_dim, n_heads, n_kv_heads, &device);
        let rope = RoPE::new(head_dim, 128, 10000.0, &device);

        // 1. Full Sequence Forward (Standard)
        let seq_len = 4;
        let input = Tensor::<3>::random(
            [1, seq_len, hidden_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        // Standard forward (no cache)
        let out_std = attn.forward(input.clone(), &rope, None, None);

        // 2. Step-by-Step Forward (Paged Cache)
        let mut cache = KVCache::new();
        let mut out_paged_parts = Vec::new();

        for i in 0..seq_len {
            // Check output of step i
            let input_step = input.clone().slice([0..1, i..i + 1, 0..hidden_dim]);

            // Forward with cache
            let out_step = attn.forward(input_step, &rope, None, Some(&mut cache));
            out_paged_parts.push(out_step);
        }

        let out_paged = Tensor::cat(out_paged_parts, 1);

        // 3. Compare
        // We expect some floating point divergence due to different calculation order (block sum vs global softmax)
        // Wait, Global Softmax over concat scores IS mathematically equivalent.
        // But in step-by-step decoding, we only attend to past!
        // In Full Sequence Standard Forward, we use Causal Mask.
        // Causal Mask ensures pos i only attends to 0..=i.
        // In Cached Forward step i, we attend to cache which contains 0..=i.
        // So they SHOULD be equivalent.

        let diff = (out_std - out_paged).abs().max().into_scalar::<f32>();
        println!("Max Diff: {}", diff);
        assert!(diff < 1e-4);
    }

    /// 稀疏注意力：prefill（整段 + sparse mask）与逐 token 解码（cache + sparse
    /// mask）逐行可见集一致 → 输出应一致。验证 mask 的 q_offset 语义正确。
    #[test]
    fn test_sparse_prefill_matches_sparse_decode() {
        use crate::model::sparse::SparseAttentionConfig;

        let device = get_device();
        let hidden_dim = 16;
        let n_heads = 4;
        let n_kv_heads = 2; // GQA
        let head_dim = hidden_dim / n_heads;

        let mut attn = GroupedQueryAttention::new(hidden_dim, n_heads, n_kv_heads, &device);
        // block=2：seq 8 → 4 块；每行选 top-2 历史块 ∪ 最近 2 token 窗口
        let sparse_cfg = SparseAttentionConfig::new(2, 2, 2);
        attn.set_sparse(Some(sparse_cfg));
        let rope = RoPE::new(head_dim, 128, 10000.0, &device);

        // 1. prefill：整段 + sparse mask（offset=0）
        let seq_len = 8;
        let input = Tensor::<3>::random(
            [1, seq_len, hidden_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out_prefill = attn.forward(input.clone(), &rope, None, None);

        // 2. decode：逐 token 走 cache（每步 offset = 已缓存长度）
        let mut cache = KVCache::new();
        let mut parts = Vec::new();
        for i in 0..seq_len {
            let step = input.clone().slice([0..1, i..i + 1, 0..hidden_dim]);
            let out = attn.forward(step, &rope, None, Some(&mut cache));
            parts.push(out);
        }
        let out_decode = Tensor::cat(parts, 1);

        // 3. 稀疏语义下两者可见集一致 → 数学一致
        let diff = (out_prefill - out_decode).abs().max().into_scalar::<f32>();
        println!("Sparse prefill vs decode Max Diff: {}", diff);
        assert!(diff < 1e-4);
    }

    /// 稀疏 + 页路径端到端：`block_size == 16 == cache 页大小` → decode 每步
    /// 走 sparse_exec 页路径（head_rows 打分 + gather_spans），仍与 prefill
    /// 的 mask 语义逐行一致。
    #[test]
    fn test_sparse_decode_paged_path_matches_prefill() {
        use crate::model::sparse::SparseAttentionConfig;

        let device = get_device();
        let hidden_dim = 16;
        let n_heads = 4;
        let n_kv_heads = 2; // GQA
        let head_dim = hidden_dim / n_heads;

        let mut attn = GroupedQueryAttention::new(hidden_dim, n_heads, n_kv_heads, &device);
        // bs == cache 页大小（DEFAULT_PAGE_SIZE = 16）→ 解码启用页读取
        let sparse_cfg = SparseAttentionConfig::new(16, 2, 16);
        attn.set_sparse(Some(sparse_cfg));
        let rope = RoPE::new(head_dim, 128, 10000.0, &device);

        // prefill：整段 + sparse mask（offset=0）
        let seq_len = 40; // → 3 页（16,16,8），跨页与末页偏置都覆盖
        let input = Tensor::<3>::random(
            [1, seq_len, hidden_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out_prefill = attn.forward(input.clone(), &rope, None, None);

        // decode：逐 token（每步走页路径）
        let mut cache = KVCache::new();
        let mut parts = Vec::new();
        for i in 0..seq_len {
            let step = input.clone().slice([0..1, i..i + 1, 0..hidden_dim]);
            let out = attn.forward(step, &rope, None, Some(&mut cache));
            parts.push(out);
        }
        let out_decode = Tensor::cat(parts, 1);

        let diff = (out_prefill - out_decode).abs().max().into_scalar::<f32>();
        println!("Sparse paged-path decode vs prefill Max Diff: {}", diff);
        assert!(
            diff < 1e-4,
            "页路径 decode 应与 prefill 一致，实际 diff = {diff}"
        );
    }
}
