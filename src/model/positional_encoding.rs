use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct RoPE<B: Backend> {
    pub freq_cis: Tensor<B, 4>, // [1, 1, max_seq_len, head_dim / 2]
}

impl<B: Backend> RoPE<B> {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32, device: &B::Device) -> Self {
        // 计算旋转频率 theta_i = theta ^ (-2i/d)
        let mut inv_freq = Vec::with_capacity(dim / 2);
        for i in (0..dim).step_by(2) {
            let freq = 1.0 / theta.powf(i as f32 / dim as f32);
            inv_freq.push(freq);
        }

        // [dim / 2]
        let inv_freq = Tensor::<B, 1>::from_data(inv_freq.as_slice(), device);

        // [max_seq_len]
        let t = Tensor::<B, 1>::from_data(
            (0..max_seq_len)
                .map(|x| x as f32)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        );

        // 外积得到频率矩阵 [max_seq_len, dim / 2]
        // 这里使用 unsqueeze 和 mul 来模拟外积
        let t = t.reshape([max_seq_len, 1]);
        let inv_freq = inv_freq.reshape([1, dim / 2]);
        let freqs = t.matmul(inv_freq);

        // 为后续广播做准备 [1, 1, max_seq_len, dim / 2]
        let freqs = freqs.reshape([1, 1, max_seq_len, dim / 2]);

        Self { freq_cis: freqs }
    }

    /// 应用 RoPE 旋转
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // x: [Batch, NumHeads, SeqLen, HeadDim]
        let dims = x.dims();
        let seq_len = dims[D - 2];
        let head_dim = dims[D - 1];

        // 1. 将 x reshape 成 [..., SeqLen, HeadDim/2, 2]
        let x_reshaped = x
            .clone()
            .reshape([dims[0], dims[1], seq_len, head_dim / 2, 2]);

        // 2. 提取实部和虚部 (模拟复数旋转)
        let x0 = x_reshaped
            .clone()
            .slice([0..dims[0], 0..dims[1], 0..seq_len, 0..(head_dim / 2), 0..1])
            .squeeze_dim(4);
        let x1 = x_reshaped
            .slice([0..dims[0], 0..dims[1], 0..seq_len, 0..(head_dim / 2), 1..2])
            .squeeze_dim(4);

        // 3. 获取当前长度的频率
        let freqs = self
            .freq_cis
            .clone()
            .slice([0..1, 0..1, 0..seq_len, 0..(head_dim / 2)]);

        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        // 4. 旋转 [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
        let rx0 = x0.clone() * cos.clone() - x1.clone() * sin.clone();
        let rx1 = x0 * sin + x1 * cos;

        // 5. 拼接并还原形状
        let rx0 = rx0.unsqueeze_dim(4);
        let rx1 = rx1.unsqueeze_dim(4);

        Tensor::<B, 5>::cat(vec![rx0, rx1], 4).reshape(dims)
    }
}
