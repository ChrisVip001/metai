use crate::model::ffn::SwiGLU;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Mixture of Experts (MoE) 模块实现
/// 支持密集共享专家与稀疏活跃专家路由
#[derive(Module, Debug)]
pub struct MoE<B: Backend> {
    pub gate: Linear<B>,
    pub experts: Vec<SwiGLU<B>>,
    pub shared_experts: Vec<SwiGLU<B>>,
    pub num_experts: usize,
    pub active_experts: usize,
}

impl<B: Backend> MoE<B> {
    pub fn new(
        hidden_dim: usize,
        mlp_dim: usize,
        num_experts: usize,
        active_experts: usize,
        num_shared_experts: usize,
        device: &B::Device,
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

        // 初始化负载均衡与路由参数
        Self {
            gate,
            experts,
            shared_experts,
            num_experts,
            active_experts,
        }
    }

    /// 优化的 MoE 前向传播 (Sparse Execution)
    ///
    /// 采用 Gather-Compute-Scatter 模式，仅对路由选中的 Token 进行专家计算。
    /// 极大减少了超大规模专家系统 (如 256 专家) 的计算负担。
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dims = x.dims();
        let batch_seq_len = dims.iter().take(D - 1).product();
        let hidden_dim = dims[D - 1];

        // 1. 展平为 [Batch * Seq, Hidden]
        let x_flat = x.reshape([batch_seq_len, hidden_dim]);

        // --- 共享专家计算 (Shared Experts) ---
        let mut shared_output = x_flat.zeros_like();
        if !self.shared_experts.is_empty() {
            for shared_expert in &self.shared_experts {
                shared_output = shared_output + shared_expert.forward(x_flat.clone());
            }
        }

        // --- 路由专家计算 (Routed Experts) ---

        // 2. 计算门控权重: [Batch * Seq, NumExperts]
        let gate_logits = self.gate.forward(x_flat.clone());

        // 3. 获取 Top-K 专家索引和权重
        let (weights, indices) = gate_logits.topk_with_indices(self.active_experts, 1);
        let weights = burn::tensor::activation::softmax(weights, 1);

        // 4. 初始化路由输出
        let mut routed_output = x_flat.zeros_like();

        // 5. 稀疏路由优化循环
        for k in 0..self.active_experts {
            let k_indices = indices
                .clone()
                .slice([0..batch_seq_len, k..(k + 1)])
                .flatten::<1>(0, 1);
            let k_weights = weights
                .clone()
                .slice([0..batch_seq_len, k..(k + 1)])
                .flatten::<1>(0, 1);

            for expert_idx in 0..self.num_experts {
                let mask = k_indices.clone().equal_elem(expert_idx as i32);

                // 获取选中该专家的 Token 索引
                let selected_indices = mask.argwhere().flatten::<1>(0, 1);
                let num_selected = selected_indices.dims()[0];

                if num_selected == 0 {
                    continue;
                }

                // Gather: 提取选中 Token
                let x_selected = x_flat.clone().select(0, selected_indices.clone());

                // Compute: 专家计算
                let expert_out = self.experts[expert_idx].forward(x_selected);

                // Weighting: 加权
                let w_selected = k_weights
                    .clone()
                    .select(0, selected_indices.clone())
                    .unsqueeze::<2>();
                let weighted_out = expert_out * w_selected;

                // Scatter: 加回主流
                let indices_expanded = selected_indices
                    .clone()
                    .reshape([num_selected, 1])
                    .expand([num_selected, hidden_dim]);

                let current_val = routed_output.clone().select(0, selected_indices.clone());
                routed_output =
                    routed_output.scatter(0, indices_expanded, current_val + weighted_out);
            }
        }

        // Final Output = Shared + Routed
        let final_output = shared_output + routed_output;

        final_output.reshape(dims)
    }
}
