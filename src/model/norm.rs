use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    pub epsilon: f64,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(dim: usize, epsilon: f64, device: &B::Device) -> Self {
        let gamma = Param::from_tensor(Tensor::ones([dim], device));
        Self { gamma, epsilon }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dim = x.dims()[D - 1];
        let eps = self.epsilon;

        // 计算方差的均值 (RMS)
        // 使用针对最后维度的方差计算
        let norm = (x.clone().powf_scalar(2.0).mean_dim(D - 1) + eps).sqrt();
        let x = x / norm;

        // 应用缩放因子
        let gamma = self.gamma.val();

        // 构造广播形状: [1, 1, ..., Dim]
        let mut shape = [1; D];
        shape[D - 1] = dim;

        x * gamma.reshape(shape)
    }
}
