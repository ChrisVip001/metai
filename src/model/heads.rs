//! # HeadOps —— 多头布局算子（Head Split / Merge / KV-Head Repeat）
//!
//! 注意力机制在数学上等价于对「头」维度的分组并行。线性投影输出的维度
//! 习惯上排列为 \(\text{hidden} = n\_heads \times head\_dim\)，而注意力计算
//! 需要把这些元素重新组织成 4 维张量 \([Batch, Heads, Seq, HeadDim]\)，
//! 使得同一次 `matmul` 可以并行计算所有头。
//!
//! ## 为什么需要这个算子
//!
//! 纯张量库（如 Burn）的 `matmul` 是批量操作：把 `[B, H, S, D] @ [B, H, D, S']`
//! 视为 \(B\times H\) 个独立矩阵乘。因此**布局变换**（head split / merge）
//! 的正确性直接决定注意力结果是否正确，而它本身又是纯「内存视图」操作
//! （`reshape` + `swap_dims`，零计算量）。把它独立出来：
//!
//! - 便于单独验证与单测（如验证 split 后再 merge 恒等）；
//! - 未来可替换为**零拷贝视图**（如只改变步长）或与上/下游 kernel 融合，
//!   避免 `reshape/transpose` 带来的数据搬运；
//! - 在量化/混合精度下统一处理布局。
//!
//! ## 关键点：reshape 的行优先语义
//!
//! Burn（与 PyTorch/Numpy 相同）采用 **row-major（行优先）** 布局：
//! 相邻元素按最后一个维度连续。因此：
//!
//! - `[B, S, H*D]` → reshape → `[B, S, H, D]` 是**逐行切分**，即每个 token 的
//!   向量被切成 `H` 段，每段长 `D`（这正是投影权重的天然布局，无需拷贝）；
//! - `[B, S, H, D]` → `swap_dims(1,2)` → `[B, H, S, D]` 才是真正的转置
//!   （把第 1、2 维交换），这一步会有数据搬运；
//! - merge 是上述过程的逆操作：`[B, H, S, D]` → `swap_dims(1,2)` →
//!   `[B, S, H, D]` → reshape → `[B, S, H*D]`。
//!
//! ## GQA / MQA：为什么需要 Repeat 算子
//!
//! - **MHA**：每个查询头有独立的 K/V 头，\(n\_kv = n\_heads\)，不需要 repeat；
//! - **MQA**（Multi-Query）：所有查询头**共享 1 组** K/V，\(n\_kv = 1\)，
//!   KV Cache 显存降到 1/\(n\_heads\)，但表达能力下降；
//! - **GQA**（Grouped-Query，本项目默认）：介于两者之间，
//!   \(n\_heads\) 个查询头被分为 \(n\_kv\) 组，每组共享 1 组 K/V
//!   （\(n\_rep = n\_heads / n\_kv\)），在**显存占用**与**表达能力**之间折中。
//!
//! GQA 的数学正确性：设每组头共享同一 K/V，则
//!
//! \[
//! \text{attn}_i(q_i, K, V) = \text{softmax}\!\left(\frac{q_i K^\top}{\sqrt{d_k}}\right)V
//! \]
//!
//! 对同组内所有 \(i\) 都成立。因此实现时先把 K/V 的每个头沿「头维」复制
//! \(n\_rep\) 份（`repeat_kv`），使形状与 Q 对齐，即可复用与 MHA 完全相同的
//! 批量矩阵乘路径。
//!
//! ## 函数清单（全部无参数，不进入 checkpoint）
//!
//! | 函数 | 形状变化 | 用途 |
//! |---|---|---|
//! | `split_heads` | `[B,S,H*D] → [B,H,S,D]` | Q / K / V 投影后拆头 |
//! | `merge_heads` | `[B,H,S,D] → [B,S,H*D]` | 注意力输出合并头，接输出投影 |
//! | `repeat_kv`   | `[B,K,S,D] → [B,K*n_rep,S,D]` | GQA 的 K/V 头扩展 |

use burn::tensor::Tensor;

/// 把单个高维向量按头切分：`[Batch, Seq, Heads*HeadDim] -> [Batch, Heads, Seq, HeadDim]`。
///
/// `x` 通常是 `w_q` / `w_k` / `w_v` 线性投影的输出（无 bias）。
/// `n_heads` 为头的数量；`head_dim` 由 `hidden / n_heads` 自动推导。
pub fn split_heads(x: Tensor<3>, n_heads: usize) -> Tensor<4> {
    let [batch_size, seq_len, hidden_dim] = x.dims();
    let head_dim = hidden_dim / n_heads;

    // 先按行优先切成 [B, S, H, D]，再交换 S 与 H 得到 [B, H, S, D]。
    x.reshape([batch_size, seq_len, n_heads, head_dim])
        .swap_dims(1, 2)
}

/// 合并多头布局回投影空间：`[Batch, Heads, Seq, HeadDim] -> [Batch, Seq, Heads*HeadDim]`。
///
/// 用于注意力输出投影（`w_o`）之前，形状信息全部从张量自身推导。
pub fn merge_heads(x: Tensor<4>) -> Tensor<3> {
    let [batch_size, n_heads, seq_len, head_dim] = x.dims();

    // 逆操作：先把 [B,H,S,D] 转回 [B,S,H,D]，再合并 H、D 两个维度。
    x.swap_dims(1, 2)
        .reshape([batch_size, seq_len, n_heads * head_dim])
}

/// GQA 的 K/V 头扩展算子：`[Batch, KVHeads, Seq, HeadDim] -> [Batch, KVHeads*n_rep, Seq, HeadDim]`。
///
/// 复制实现采用「插入单位维度 → expand(广播复制) → reshape」三步，
/// 避免真正把数据复制多份内存（由后端在读取时按步长广播）。
/// 当 `n_rep == 1`（即 MHA 或不做 GQA 时）直接返回原张量，零开销。
pub fn repeat_kv(x: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return x;
    }

    let [batch_size, n_kv_heads, seq_len, head_dim] = x.dims();

    // 1) [B, K, S, D] -> [B, K, 1, S, D]   （在 seq 维前插入单位维）
    // 2) -> [B, K, n_rep, S, D]            （expand 广播）
    // 3) -> [B, K * n_rep, S, D]           （reshape 合并 K 与 n_rep）
    x.unsqueeze_dims::<5>(&[2])
        .expand([batch_size, n_kv_heads, n_rep, seq_len, head_dim])
        .reshape([batch_size, n_kv_heads * n_rep, seq_len, head_dim])
}
