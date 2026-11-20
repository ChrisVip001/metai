//! # SDPA —— 缩放点积注意力算子（Scaled Dot-Product Attention）
//!
//! 自注意力中最核心的数学环节。输入已是多头布局的 Q / K / V：
//!
//! \[
//! \text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}} + M\right) V
//! \]
//!
//! 其中 \(Q,K,V \in \mathbb{R}^{B \times H \times S \times d_k}\)，\(M\) 为
//! 掩码（被屏蔽位置取 \(-\infty\)，softmax 后概率为 0）。本模块把这一
//! **单一数学算子**与 mask 的构造独立出来。
//!
//! ## 分步拆解
//!
//! 1. **打分**：`scores = Q @ Kᵀ`，形状 `[B, H, S_q, S_k]`。第 \((i,j)\) 个元素
//!    是第 \(i\) 个查询与第 \(j\) 个键的内积，衡量两者相关度；
//! 2. **缩放**：除以 \(\sqrt{d_k}\)。当 \(d_k\) 较大时内积方差 \(\approx d_k\)，
//!    softmax 会进入梯度极小的饱和区，除以 \(\sqrt{d_k}\) 使 logits 方差保持在
//!    O(1)，保证梯度流动；
//! 3. **掩码**：causal mask（未来不可见）与 padding mask（pad 位置不可见）
//!    以 `-inf` 填充被屏蔽位置（见 [`combine_masks`]）；
//! 4. **归一化**：对最后一维做 softmax，得到每个查询对所有键的注意力权重
//!    \(\alpha_{i,j} \ge 0,\ \sum_j \alpha_{i,j} = 1\)；
//! 5. **加权求和**：`out = probs @ V`，每个查询输出为所有 Value 的凸组合，
//!    形状回到 `[B, H, S_q, d_k]`。
//!
//! ## 复杂度
//!
//! - 计算：\(O(B \cdot H \cdot S^2 \cdot d_k)\)——与序列长度**平方**成正比，
//!   是长上下文场景的主要瓶颈；
//! - 显存：仅 `scores` 一项就需要保存 \(B \cdot H \cdot S^2\) 个元素
//!   （反向传播需要它），这就是为什么会出现 FlashAttention 等 kernel。
//!
//! ## 数值稳定性
//!
//! naive softmax 直接计算 \(e^{x}\) 会溢出（`x` 大时 `e^x -> inf`）。Burn 的
//! `softmax` 内部实现了 *max-subtraction*（先减最大值再求幂）以保证数值稳定。
//! 但在自定义 kernel 时需自行处理，且**必须避免**在 mask 位置产生 NaN：
//! `-inf + inf` 是 NaN，因此常规实现先把 mask 位置填 `-inf` 再 softmax
//! （本模块即如此），或采用 *masked softmax* 专用 kernel。
//!
//! ## 可独立优化的方向（本算子即替换点）
//!
//! - **FlashAttention / Online Softmax**：不物化整个 `scores`，按块扫描并
//!   维护 running max/sum，把复杂度（显存）降到 \(O(S)\)，同时融合 mask；
//! - **PagedAttention**：K/V 分块存储，仅在解码时计算当前块与历史块的
//!   部分得分（本项目 `infer::cache` 预留了该方向）；
//! - 上三角 mask 预计算并按块复用；`int8/fp8` 量化 QKᵀ；MLA 低秩压缩等。
//!
//! ## 形状约束
//!
//! - 输入：`q,k,v` 均为 `[Batch, Heads, Seq, HeadDim]`（Q 与 K/V 的 Seq 可不同，
//!   如解码时 Q 只有 1 行而 K/V 来自 Cache）；
//! - `mask`：`[Batch(或1), 1(或Heads), S_q, S_k]` 的 Bool 张量，`true` 表示屏蔽；
//! - 输出：`[Batch, Heads, S_q, HeadDim]`。

use burn::tensor::activation::softmax;
use burn::tensor::{Bool, Tensor};

/// 缩放点积注意力核心算子（无参数）。
///
/// 流程：`scores = Q·Kᵀ / √d_k` → 可选 mask 填 `-inf` → softmax(最后一维) →
/// `probs · V`。
///
/// - `q / k / v`：已处于 `[Batch, Heads, Seq, HeadDim]` 多头布局
///   （见 [`crate::model::heads`]）；
/// - `scale`：缩放因子，通常为 `1 / sqrt(head_dim)`；
/// - `mask`：4 维 Bool 掩码，`true` = 屏蔽（填 `-inf`）。已支持广播
///   （例如 `[1, 1, S, S]` 对任意 batch / heads 生效）。
pub fn scaled_dot_product_attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scale: f32,
    mask: Option<Tensor<4, Bool>>,
) -> Tensor<4> {
    // scores: [B, H, S_q, S_k] = Q @ Kᵀ
    let scores = q.matmul(k.swap_dims(2, 3)) * scale;
    let probs = if let Some(m) = mask {
        // Rows with no visible key contribute zero instead of softmax(-inf, ...).
        softmax(scores.mask_fill(m.clone(), -1.0e30), 3).mask_fill(m, 0.)
    } else {
        softmax(scores, 3)
    };

    // 加权求和：out = probs @ V，[B, H, S_q, HeadDim]
    probs.matmul(v)
}

/// 构建因果（causal / 下三角）掩码。
///
/// 返回 `[S, S]` 的 Bool 张量，`true` 表示「该位置被屏蔽」。
/// 规则：第 \(i\) 行（查询 \(i\)）只能看到第 \(0..=i\) 列（键），
/// 即上三角（含主对角线右上）为 `true`。实现用 `ones.triu(1)`：
/// 主对角线以上（偏移 +1）取 1。
///
/// ```text
///   键 j →      0    1    2    3
/// 查询 i ↓
///    0       [false, true, true, true]
///    1       [false,false, true, true]
///    2       [false,false,false, true]
///    3       [false,false,false,false]
/// ```
pub fn build_causal_mask(seq_len: usize, device: &burn::tensor::Device) -> Tensor<2, Bool> {
    Tensor::<2>::ones([seq_len, seq_len], device).triu(1).bool()
}

/// 将「因果掩码」与「padding 掩码」合并为一张 4 维 Bool 掩码（逻辑或）。
///
/// - `causal`：建议为 `[1, 1, S, S]`（或 `[S, S]` 已自行扩展）；
/// - `padding`：`[B, 1, 1, S]`，其中 `true` 表示该 token 是 pad（需屏蔽）；
/// - 返回 `[B, 1, S, S]`（由 `batch_size`/`seq_len` 参数广播而来），
///   `true` = causal 屏蔽 **或** padding 屏蔽。
///
/// 合并手法：把 Bool 转 `Int`（0/1）相加，`>= 1` 即为逻辑或，可避免依赖
/// 后端对 Bool 逻辑运算的支持。
pub fn combine_masks(
    causal: Tensor<4, Bool>,
    padding: Tensor<4, Bool>,
    batch_size: usize,
    seq_len: usize,
) -> Tensor<4, Bool> {
    let causal = causal.expand([batch_size, 1, seq_len, seq_len]);
    let padding = padding.expand([batch_size, 1, seq_len, seq_len]);

    (causal.int() + padding.int()).greater_equal_elem(1)
}

/// Offset-aware causal mask for cached multi-token prefill and verification.
pub fn cached_mask(
    offset: usize,
    n: usize,
    device: &burn::tensor::Device,
    padding: Option<Tensor<4, Bool>>,
) -> Option<Tensor<4, Bool>> {
    if n == 1 {
        return padding;
    }
    use burn::tensor::Int;
    let len = offset + n;
    let keys = Tensor::<1, Int>::arange(0..len as i64, device).reshape([1, 1, 1, len]);
    let queries = Tensor::<1, Int>::arange(offset as i64..len as i64, device).reshape([1, 1, n, 1]);
    let future = keys.greater(queries);
    Some(match padding {
        Some(p) => future.bool_or(p),
        None => future,
    })
}
