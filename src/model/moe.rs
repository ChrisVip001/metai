//! # MoE —— 混合专家组合算子（Mixture of Experts，稀疏前馈层）
//!
//! 现代大模型（如 DeepSeek-V3 / Mixtral）用 MoE 替换稠密 FFN：
//! 总参数量（容量）巨大，但每个 token **只经过少量被路由选中的专家**，
//! 从而把单 token 的计算量（FLOPs）控制在与稠密模型相当的水平。
//!
//! \[
//! \text{MoE}(x) = \underbrace{\sum_{e \in \text{shared}} \mathrm{FFN}_e(x)}_{\text{共享专家（总是激活）}}
//!               + \underbrace{\sum_{e \in \text{TopK}(g(x))} w_e \cdot \mathrm{FFN}_e(x)}_{\text{路由专家（稀疏激活）}}
//! \]
//!
//! ## 三个组成部分（各自独立）
//!
//! 1. **路由算子**（[`crate::model::router`]）：`gate(x)=x·W_g` 打分 →
//!    Top-K 选专家 → softmax 归一化 → 返回 `(weights, indices)`；
//! 2. **专家网络**（[`crate::model::ffn::SwiGLU`]）：每个专家是一个完整
//!    的 SwiGLU FFN（本实现为稠密 shared + 稀疏 routed 两种角色）；
//! 3. **稀疏聚合**（本文件 forward）：按 `indices` 把 token 分发（Gather）到
//!    对应专家计算（Compute）、按 `weights` 加权并写回（Scatter）。
//!
//! ## 共享专家（Shared Experts）—— DeepSeek-V3 风格
//!
//! 共享专家**对每个 token 无条件激活**，其作用是承载「所有 token 都需要」
//! 的公共变换。这样路由专家可以更专精、路由头数可以更少。
//! 本项目配置里 `num_shared_experts` 决定其个数（0 表示关闭）。
//!
//! ## Gather-Compute-Scatter 稀疏执行（本 forward 的核心）
//!
//! 朴素实现会遍历**所有** \(E\) 个专家并计算每个 token，复杂度
//! \(O(T·E)\)。稀疏执行只对「被路由选中」的 token 调专家：
//!
//! ```text
//! 对第 k 个选中槽位（k = 0..K）：
//!   索引向量 idx_k ∈ [T]          ← TopK 结果的第 k 列
//!   对每个专家 e（e = 0..E）：
//!     mask     = (idx_k == e)                  ← 谁选了专家 e
//!     selected = argwhere(mask)                ← 这些 token 的下标
//!     x_e      = gather(x_flat, selected)      ← Gather：取回 token
//!     y_e      = expert_e(x_e)                 ← Compute：专家计算
//!     y_e     *= w_k[selected]                 ← Weighting：路由权重
//!     out      = scatter(out, selected, y_e)   ← Scatter：写回主流
//! ```
//!
//! 平均每个 token 只触发 \(K\) 次专家计算，训练/推理开销
//! \(O(T·K)\)（另加 shared 专家的 \(O(T)\)），远小于 \(O(T·E)\)。
//!
//! ## 关键工程注意点
//!
//! - 稀疏分发后每个 token 只被一个专家处理（此处是互斥 Top-K 槽位），
//!   **无容量限制**（capacity factor = ∞），简单但极端长尾下负载不均；
//! - 若要限制容量（如 `capacity = ⌈T/E⌉·C`）或加入负载均衡辅助损失，
//!   可在 [`crate::model::router`] 与聚合循环处扩展（详见 router 模块 doc）；
//! - 当 token 数稀疏时专家分支多为空（`num_selected == 0` 直接跳过，
//!   避免无意义的 kernel launch）。
//!
//! ## 形状约束
//!
//! - 输入：任意 `[.., Hidden]` 形状，内部展平为 `[T, Hidden]`（T = batch×seq）；
//! - gate 打分：`[T, E]`；Top-K：`weights [T, K]`、`indices [T, K]`；
//! - 输出：恢复输入原形状 `[.., Hidden]`。

use crate::model::ffn::SwiGLU;
use crate::model::router::{self, RouterVariant};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;

/// MoE 组合算子：路由 + 稀疏专家 + 共享专家 + 加权聚合。
///
/// 结构字段（保持稳定，checkpoint 权重路径依赖这些字段名）：
/// - `gate`：路由打分线性层 `[num_experts, hidden]`（逻辑见 router 算子）；
/// - `experts`：被路由稀疏激活的专家集合（SwiGLU）；
/// - `shared_experts`：对每个 token 都激活的共享专家集合（可为空）；
/// - `num_experts` / `active_experts`：专家总数 / 每个 token 激活数；
/// - `router_variant`：路由算法变体（非权重、不参与梯度，见 [`RouterVariant`]；
///   用 `Ignored` 容器声明以符合 burn `Module` 派生约束）。
#[derive(Module, Debug)]
pub struct MoE {
    pub gate: Linear,
    pub experts: Vec<SwiGLU>,
    pub shared_experts: Vec<SwiGLU>,
    pub num_experts: usize,
    pub active_experts: usize,
    #[module(skip)]
    pub router_variant: RouterVariant,
}

impl MoE {
    /// 构造 MoE。
    ///
    /// - `hidden_dim`：输入/输出维度；
    /// - `mlp_dim`：单个专家的中间维度；
    /// - `num_experts`：路由专家总数；
    /// - `active_experts`：每个 token 激活的路由专家数（Top-K）；
    /// - `num_shared_experts`：共享专家个数（0 = 不启用）；
    /// - `device`：权重所在设备。
    pub fn new(
        hidden_dim: usize,
        mlp_dim: usize,
        num_experts: usize,
        active_experts: usize,
        num_shared_experts: usize,
        device: &burn::tensor::Device,
    ) -> Self {
        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts.push(SwiGLU::new(hidden_dim, mlp_dim, device));
        }

        let mut shared_experts = Vec::with_capacity(num_shared_experts);
        for _ in 0..num_shared_experts {
            shared_experts.push(SwiGLU::new(hidden_dim, mlp_dim, device));
        }

        let gate = LinearConfig::new(hidden_dim, num_experts)
            .with_bias(false)
            .init(device);

        Self {
            gate,
            experts,
            shared_experts,
            num_experts,
            active_experts,
            router_variant: RouterVariant::SoftmaxTopK,
        }
    }

    /// 设定路由算法变体（组合层装配时调用；默认 softmax top-K）。
    pub fn set_router(&mut self, variant: RouterVariant) {
        self.router_variant = variant;
    }

    /// MoE 前向传播（Gather-Compute-Scatter 稀疏执行）。
    ///
    /// 仅对路由选中的 token 进行专家计算，避免遍历全部专家，
    /// 极大减少超大规模专家系统（如 256 专家）的计算负担。
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let dims = x.dims();
        let batch_seq_len = dims.iter().take(D - 1).product();
        let hidden_dim = dims[D - 1];

        // 1. 展平为 [Batch*Seq, Hidden]，便于按 token 做行级路由与分发
        let x_flat = x.reshape([batch_seq_len, hidden_dim]);

        // With one softmax expert every row selects index zero. Keep the same
        // gate and gradients, but avoid the GPU -> host argwhere round trip.
        if self.num_experts == 1
            && self.active_experts == 1
            && self.shared_experts.is_empty()
            && self.router_variant == RouterVariant::SoftmaxTopK
        {
            let weights = burn::tensor::activation::softmax(self.gate.forward(x_flat.clone()), 1);
            return (self.experts[0].forward(x_flat) * weights).reshape(dims);
        }

        // ===== 分支 A：共享专家（对每个 token 无条件激活并累加） =====
        let mut shared_output = x_flat.zeros_like();
        if !self.shared_experts.is_empty() {
            for shared_expert in &self.shared_experts {
                shared_output = shared_output + shared_expert.forward(x_flat.clone());
            }
        }

        // ===== 分支 B：路由专家（稀疏激活） =====

        // 2. 路由决策（独立算子 router::route_with，按 router_variant 分发）：
        //    返回 weights [T, K]（归一化后）与 indices [T, K]（选中的专家号）
        let (weights, indices) = router::route_with(
            x_flat.clone(),
            &self.gate,
            self.active_experts,
            self.router_variant,
        );

        // 3. 初始化路由输出缓冲
        let mut routed_output = x_flat.zeros_like();

        // 4. 稀疏聚合：逐 Top-K 槽位分发（Gather → Compute → Weighting → Scatter）
        //    该循环在未来可被替换为单 kernel 的 Grouped-GEMM + 稀疏 Scatter。
        for k in 0..self.active_experts {
            // 第 k 个选中槽位：每个 token 对应的专家索引与权重 [T]
            let k_indices = indices
                .clone()
                .slice([0..batch_seq_len, k..(k + 1)])
                .flatten::<1>(0, 1);
            let k_weights = weights
                .clone()
                .slice([0..batch_seq_len, k..(k + 1)])
                .flatten::<1>(0, 1);

            for (expert_idx, expert) in self.experts.iter().enumerate() {
                // 找出「第 k 槽选中了专家 expert_idx」的 token
                let mask = k_indices.clone().equal_elem(expert_idx as i32);
                let selected_indices = mask.argwhere().flatten::<1>(0, 1);
                let num_selected = selected_indices.dims()[0];

                // 该专家在本批没有被任何 token 选中 → 跳过（避免空 kernel）
                if num_selected == 0 {
                    continue;
                }

                // Gather：按索引收集被选中 token 的特征 [N, Hidden]
                let x_selected = x_flat.clone().select(0, selected_indices.clone());

                // Compute：被选中 token 走该专家网络 [N, Hidden]
                let expert_out = expert.forward(x_selected);

                // Weighting：乘上第 k 槽的路由权重（softmax 系数）[N, 1]
                let w_selected = k_weights
                    .clone()
                    .select(0, selected_indices.clone())
                    .reshape([num_selected, 1]);
                let weighted_out = expert_out * w_selected;

                // Scatter：把加权结果写回输出流对应行（累加）
                let indices_expanded = selected_indices
                    .clone()
                    .reshape([num_selected, 1])
                    .expand([num_selected, hidden_dim]);

                routed_output = routed_output.scatter(
                    0,
                    indices_expanded,
                    weighted_out,
                    burn::tensor::IndexingUpdateOp::Add,
                );
            }
        }

        // 5. 最终输出 = 共享专家结果 + 路由专家结果，恢复原形状
        let final_output = shared_output + routed_output;

        final_output.reshape(dims)
    }
}
