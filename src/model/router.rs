//! # MoE Router —— 专家路由决策算子（Token → Expert Routing）
//!
//! MoE（Mixture of Experts）的核心思想是：**对一个 token 只激活一小部分
//! 专家网络**，从而在保持模型容量（参数总量）不变的前提下，大幅降低每个
//! token 的实际计算量（FLOPs）。
//!
//! 路由决策（Routing）由「门控网络」完成，本模块将其抽成独立算子：
//!
//! ```text
//! token 特征 x  ──►  gate(x) = x·W_g        （线性打分，E 个专家各一个分）
//!                    │
//!                    ▼
//!              Top-K 选择（取分最高的 K 个专家）
//!                    │
//!                    ▼
//!              softmax 归一化 → (权重, 索引)
//! ```
//!
//! ## 分步原理
//!
//! 1. **打分**：\(s_j = W_g x \in \mathbb{R}^{E}\)，\(E\) 为专家总数。
//!    分值表示该 token 与第 \(j\) 个专家的亲和度；
//! 2. **Top-K**：只保留得分最高的 \(K\) 个专家（\(K\) 通常为 1~8）。
//!    `topk_with_indices(k, dim=1)` 同时返回这 \(K\) 个分数与其索引；
//! 3. **归一化**：对选出的 \(K\) 个分数做 softmax，得到加权系数
//!    \(w_i \ge 0, \sum w_i = 1\)。最终输出为该 token 在选中专家上的
//!    加权结果（加权在 MoE 的聚合阶段完成，见 `crate::model::moe`）。
//!
//! ## 为什么不能简单用全量 softmax 决定路由
//!
//! 若对全部 \(E\) 个分数做 softmax 再取 top-k，权重会**稀释**：softmax 的
//! 概率和恒为 1，未选中的专家会“偷走”一部分概率质量，导致选中的专家
//! 权重偏小。因此标准做法是 **先 top-k、后 softmax**（只归一化被选中的部分），
//! 本项目即如此。DeepSeek-V3 更进一步使用 **无 softmax 的路由**
//! （`sigmoid(x)` 逐元素，乘以归一化因子 \(1/\text{count}\)），
//! 允许专家并行度更高，可作为本算子的升级选项。
//!
//! ## 负载均衡（升级点）
//!
//! 纯贪婪路由会导致「赢者通吃」：少数专家承担绝大多数 token（load
//! imbalance），使部分专家欠训练。业界主流缓解手段：
//!
//! - **辅助均衡损失（auxiliary loss）**：惩罚 \(f_j \cdot P_j\)（第 \(j\) 个
//!   专家的被选频率 × 平均路由概率），让路由分布尽量均匀；
//! - **专家容量（capacity factor）**：每个专家每个 token 批最多处理
//!   `capacity = ceil(tokens_per_batch / E) × C` 个 token，溢出则丢弃；
//! - **Router z-loss / 负载均衡 token 丢弃**等正则手段（DeepSeek-V3）。
//!
//! 这些逻辑可以独立加在本模块（保持 `gate` 权重存储不变即可），
//! 也是把路由与专家计算解耦的价值所在。
//!
//! ## 函数契约
//!
//! - 输入：`x` 为 `[Tokens, Hidden]` 的 token 特征（通常已展平 batch×seq）；
//!   `gate` 为门控线性层 `[Hidden, E]`；
//! - 输出：`weights` `[Tokens, K]`（softmax 后），`indices` `[Tokens, K]` Int
//!   （每个 token 选中的专家编号，注意 K 列之间**无序**，仅按分数排序）。

use burn::nn::Linear;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{Int, Tensor};
use serde::{Deserialize, Serialize};

/// 路由变体标识（配置序列化用，默认 `SoftmaxTopK` 保持现状行为）。
///
/// - `SoftmaxTopK`：经典 top-K + 被选分数内 softmax（Mixtral 等）；
/// - `SigmoidTopK`：`sigmoid` 打分 + top-K（DeepSeek-V3 风格，
///   无 softmax 稀释、对大批量更稳；完整版还带每专家可学 bias 与
///   count 归一化因子，见模块 doc，列为后续升级点）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RouterVariant {
    #[default]
    SoftmaxTopK,
    SigmoidTopK,
}

/// 路由决策统一入口：按 `variant` 分发到具体实现。
///
/// 返回 `(weights, indices)`：
/// - `weights` `[Tokens, K]`：选中专家的归一化权重（与顺序对应）；
/// - `indices` `[Tokens, K]`：选中专家的索引（Int），与 `weights` 同位置对应。
///
/// 调用方（MoE 聚合）将依据 `indices` 把 token 分发到对应专家计算，
/// 并按 `weights` 加权合并回主流。
pub fn route_with(
    x: Tensor<2>,
    gate: &Linear,
    active_experts: usize,
    variant: RouterVariant,
) -> (Tensor<2>, Tensor<2, Int>) {
    match variant {
        RouterVariant::SoftmaxTopK => route_softmax(x, gate, active_experts),
        RouterVariant::SigmoidTopK => route_sigmoid(x, gate, active_experts),
    }
}

/// 经典 softmax top-K 路由：`sigmoid` 前的打分形式（默认实现）。
///
/// 1) `s = gate(x) ∈ [Tokens, E]`；
/// 2) 取分最高的 K 个（含索引）；
/// 3) 仅对被选中的 K 个分数 softmax（避免概率质量被未选专家稀释）。
pub fn route(x: Tensor<2>, gate: &Linear, active_experts: usize) -> (Tensor<2>, Tensor<2, Int>) {
    route_softmax(x, gate, active_experts)
}

fn route_softmax(
    x: Tensor<2>,
    gate: &Linear,
    active_experts: usize,
) -> (Tensor<2>, Tensor<2, Int>) {
    // 1) 打分：[Tokens, E]
    let gate_logits = gate.forward(x);

    // 2) Top-K 选择：[Tokens, K] 权重 + [Tokens, K] 索引
    // Softmax of a single selected logit would be the constant 1 and freeze the gate.
    if active_experts == 1 {
        return softmax(gate_logits, 1).topk_with_indices(1, 1);
    }
    let (weights, indices) = gate_logits.topk_with_indices(active_experts, 1);

    // 3) 对选中的 K 个分数 softmax 归一化（不稀释给未选专家）
    let weights = softmax(weights, 1);

    (weights, indices)
}

/// Sigmoid top-K 路由（DeepSeek-V3 风格，无 softmax 稀释）。
///
/// 1) `s = sigmoid(gate(x)) ∈ (0,1)`：逐元素独立概率化，各专家分数互不影响；
/// 2) 取 sigmoid 分最高的 K 个专家；
/// 3) 对被选 K 个分数做 token 内归一化（\(w_i = s_i / \sum_{j \in \mathrm{topk}} s_j\)），
///    保证聚合后仍是凸组合。DeepSeek-V3 在生产中还叠加每专家可学 bias 与
///    count 归一化以实现「无 aux-loss 负载均衡」，见模块 doc 的升级点。
fn route_sigmoid(
    x: Tensor<2>,
    gate: &Linear,
    active_experts: usize,
) -> (Tensor<2>, Tensor<2, Int>) {
    // 1) sigmoid 打分：[Tokens, E]（逐元素独立，∈(0,1)）
    let scores = sigmoid(gate.forward(x));

    // 2) Top-K 选择（按 sigmoid 分）：
    let (selected, indices) = scores.topk_with_indices(active_experts, 1);

    // 3) token 内归一化：w = s / Σ s（保持凸组合）
    if active_experts == 1 {
        return (selected, indices);
    }
    let sum = selected.clone().sum_dim(1); // [Tokens, 1]（保留维）
    let weights = selected / sum;

    (weights, indices)
}
