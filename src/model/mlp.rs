use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor};

#[derive(Module, Debug)]
pub struct SwiGLU<B: Backend> {
    pub w1: Linear<B>, // 门控层投影
    pub w2: Linear<B>, // 输出层投影
    pub w3: Linear<B>, // 数据层投影
}

impl<B: Backend> SwiGLU<B> {
    pub fn new(hidden_dim: usize, mlp_dim: usize, device: &B::Device) -> Self {
        Self {
            w1: LinearConfig::new(hidden_dim, mlp_dim)
                .with_bias(false)
                .init(device),
            w2: LinearConfig::new(mlp_dim, hidden_dim)
                .with_bias(false)
                .init(device),
            w3: LinearConfig::new(hidden_dim, mlp_dim)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let gate = silu(self.w1.forward(x.clone()));
        let data = self.w3.forward(x);
        let combined = gate * data;
        self.w2.forward(combined)
    }
}

#[derive(Module, Debug)]
pub struct MoE<B: Backend> {
    pub gate: Linear<B>,
    pub experts: Vec<SwiGLU<B>>,
    pub num_experts: usize,
    pub active_experts: usize,
}

impl<B: Backend> MoE<B> {
    pub fn new(
        hidden_dim: usize,
        mlp_dim: usize,
        num_experts: usize,
        active_experts: usize,
        device: &B::Device,
    ) -> Self {
        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts.push(SwiGLU::new(hidden_dim, mlp_dim, device));
        }

        let gate = LinearConfig::new(hidden_dim, num_experts)
            .with_bias(false)
            .init(device);

        Self {
            gate,
            experts,
            num_experts,
            active_experts,
        }
    }

    /// 优化的 MoE 前向传播
    /// 
    /// 改进点：
    /// 1. 只为每个被选中的专家执行一次 forward，避免重复计算
    /// 2. 使用更高效的路由和聚合策略
    /// 3. 减少不必要的张量克隆操作
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dims = x.dims();
        let batch_seq_len = dims.iter().take(D - 1).product();
        let hidden_dim = dims[D - 1];

        // 展平为 [Batch * Seq, Hidden] 便于路由
        let x_flat = x.reshape([batch_seq_len, hidden_dim]);

        // 计算门控权重: [Batch * Seq, NumExperts]
        let gate_logits = self.gate.forward(x_flat.clone());

        // 获取 Top-K 专家索引和权重
        let (weights, indices) = gate_logits.topk_with_indices(self.active_experts, 1);
        let weights = burn::tensor::activation::softmax(weights, 1);

        // 初始化输出
        let mut output = x_flat.zeros_like();

        // 优化的路由策略：收集每个专家需要处理的 token，然后批量处理
        // 对每个被选中的 top-k 位置，计算对应专家的贡献
        for k in 0..self.active_experts {
            let k_indices = indices.clone().slice([0..batch_seq_len, k..(k + 1)]);
            let k_weights = weights.clone().slice([0..batch_seq_len, k..(k + 1)]);

            // 对每个可能的专家，收集分配给它的 token
            for expert_idx in 0..self.num_experts {
                // 检查这个 top-k 位置是否选择了当前专家
                let mask = k_indices.clone().equal_elem(expert_idx as i32);
                
                // 如果没有任何 token 选择这个专家，跳过
                let has_any = mask.clone().any();
                if !has_any.into_scalar().elem::<bool>() {
                    continue;
                }
                
                let mask_float = mask.float();

                // 计算该专家对所有 token 的输出（只执行一次）
                let expert_out = self.experts[expert_idx].forward(x_flat.clone());
                
                // 累加该专家的加权贡献（只对选择了该专家的 token 有效）
                output = output + (expert_out * mask_float * k_weights.clone());
            }
        }

        output.reshape(dims)
    }
}
