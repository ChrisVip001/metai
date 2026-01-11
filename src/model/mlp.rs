use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

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
    /// 优化的 MoE 前向传播 (Sparse Execution)
    ///
    /// 改进点：
    /// 1. 使用 Sparse Gather-Compute-Scatter 模式
    /// 2. 避免了对未选中 Expert 的 Token 进行无效计算
    /// 3. 使用 `argwhere` + `select` 提取 Token，大幅减少 FLOPs
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dims = x.dims();
        let batch_seq_len = dims.iter().take(D - 1).product();
        let hidden_dim = dims[D - 1];

        // 1. 展平为 [Batch * Seq, Hidden]
        let x_flat = x.reshape([batch_seq_len, hidden_dim]);

        // 2. 计算门控权重: [Batch * Seq, NumExperts]
        let gate_logits = self.gate.forward(x_flat.clone());

        // 3. 获取 Top-K 专家索引和权重
        let (weights, indices) = gate_logits.topk_with_indices(self.active_experts, 1);
        let weights = burn::tensor::activation::softmax(weights, 1);

        // 4. 初始化输出
        let mut output = x_flat.zeros_like();

        // 5. 稀疏路由：只计算选中的 Token
        for k in 0..self.active_experts {
            // 获取第 k 个选择的专家索引: [Batch * Seq]
            let k_indices = indices
                .clone()
                .slice([0..batch_seq_len, k..(k + 1)])
                .flatten::<1>(0, 1);
            let k_weights = weights
                .clone()
                .slice([0..batch_seq_len, k..(k + 1)])
                .flatten::<1>(0, 1);

            // 遍历所有专家 (虽然有循环，但内部计算是稀疏的)
            // 只有当 Expert 被至少一个 Token 选中时才会触发计算
            for expert_idx in 0..self.num_experts {
                // mask: [Batch * Seq]
                let mask = k_indices.clone().equal_elem(expert_idx as i32);

                // 快速跳过无人选中的 Expert (避免 Synchronization if possible, but argwhere implies sync or dynamic shape)
                // 这里我们先做一个 cheap check 或者直接 argwhere
                // 注意：在某些后端 argwhere 可能导致同步，但在 Eager 模式下没问题。
                // 如果后端支持动态形状，这是最高效的。

                // 获取选中该专家的 Token 索引
                // mask 是 1D, argwhere 返回 [NumSelected, 1]
                let selected_indices = mask.argwhere().flatten::<1>(0, 1);
                let num_selected = selected_indices.dims()[0];

                if num_selected == 0 {
                    continue;
                }

                // Gather: 提取对应 Token 的输入 [NumSelected, Hidden]
                let x_selected = x_flat.clone().select(0, selected_indices.clone());

                // Compute: 只计算选中的 Token
                let expert_out = self.experts[expert_idx].forward(x_selected);

                // Weighting: 提取对应的权重并加权
                let w_selected = k_weights
                    .clone()
                    .select(0, selected_indices.clone())
                    .unsqueeze::<2>();
                let weighted_out = expert_out * w_selected;

                // Scatter: 将结果加回主输出流
                // update = current + new_part
                // output[indices] += weighted_out
                // Burn 目前的 scatter 是替换操作，所以我们需要先 gather, add, 然后 scatter
                // 或者如果 scatter_add 可用... Burn 0.13 有 scatter_add 吗?
                // 如果没有，我们使用: output = output.scatter(indices, output.select(indices) + weighted_out)

                let [num_selected, hidden] = weighted_out.dims();
                let indices_expanded = selected_indices
                    .clone()
                    .reshape([num_selected, 1])
                    .expand([num_selected, hidden]);

                let current_val = output.clone().select(0, selected_indices.clone());
                output = output.scatter(0, indices_expanded, current_val + weighted_out);
            }
        }

        output.reshape(dims)
    }
}
