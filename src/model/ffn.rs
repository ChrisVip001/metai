use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// SwiGLU 激活函数与前馈网络实现
/// 广泛用于 Llama 3, DeepSeek 等主流 LLM 架构
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
