//! # RoPE —— 旋转位置编码算子（Rotary Position Embedding）
//!
//! 自注意力是「置换等变」的：若不打位置信息，`"A loves B"` 与 `"B loves A"`
//! 的表示完全相同。RoPE 通过在**注意力打分之前**旋转 Q / K 向量，把「相对
//! 位置」编码进注意力分数本身。
//!
//! ## 核心思想：二维平面上的旋转
//!
//! 把 \(d_k\) 维向量切成 \(d_k/2\) 个相邻对 \((x_{2i}, x_{2i+1})\)，每个对看作
//! 复平面上的一个点 \(x_{2i} + \mathrm{i}x_{2i+1}\)。对位置 \(m\) 的 token，
//! 将其第 \(i\) 对按角度 \(m\theta_i\) 旋转：
//!
//! \[
//! \begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} =
//! \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\
//!                  \sin m\theta_i &  \cos m\theta_i \end{pmatrix}
//! \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
//! \]
//!
//! 频率 \(\theta_i = \theta^{-2i/d_k}\)（本项目 \(\theta = 10000\)，与
//! 原论文一致）随维度递增而**指数衰减**：低频维度（靠前的 pair）旋转慢、
//! 编码长距离相对位置；高频维度旋转快、编码相邻 token 的细微位置差。
//!
//! ## 相对位置是怎么来的（关键性质）
//!
//! 旋转是线性正交变换，于是两个位置 \(m, n\) 的 Q/K 内积满足：
//!
//! \[
//! \langle R(m)q,\ R(n)k\rangle = \langle q,\ R(n-m)k\rangle
//! \]
//!
//! 即内积只依赖 **相对偏移** \(n-m\)，与绝对位置无关——这正是注意力打分
//! 需要的平移不变性。相比绝对位置嵌入（加法）与 ALiBi 等方案，RoPE 把
//! 位置信息直接织入打分函数，且无需在 embedding 层修改输入。
//!
//! ## 实现细节（预计算 + 分块旋转）
//!
//! 1. 构造时预计算全部频率矩阵：
//!    `freqs[m][i] = m · θ_i`，形状 `[1, 1, max_seq_len, d_k/2]`
//!    （本实现存为 `freq_cis`，注意名字沿袭复数表示，实际存的是角度）；
//! 2. 前向时按需 slice 出 `[1, 1, seq_len, d_k/2]`，求 `cos/sin`；
//! 3. 把输入 `[B, H, S, d_k]` reshape 为 `[B, H, S, d_k/2, 2]`，切出
//!    `x0`/`x1`（等价于实部/虚部）；
//! 4. 按旋转公式更新：`rx0 = x0·cos − x1·sin`，`rx1 = x0·sin + x1·cos`；
//! 5. 逐对拼接还原形状。
//!
//! ## KV Cache 配合：offset 参数
//!
//! 解码阶段每步只处理 1 个新 token（位置 = 缓存已存长度 `offset`）。RoPE
//! 必须按 token 的**真实全局位置**旋转，因此前向需传 `offset`，
//! 从预计算表中 slice `[offset, offset+seq_len)` 对应频率。这也是
//! `forward_with_offset` 存在的意义（见 [`RoPE::forward_with_offset`]）。
//!
//! ## 可独立优化方向（本算子即替换点）
//!
//! - **外推/插值**：训练长度内有效，超长会退化。可通过修改频率实现
//!   NTK-aware / YaRN / 位置插值（PI）支持 128K+ 上下文；
//! - **逐对旋转融合**：`cos/sin` 与向量乘可在同一 kernel 内完成，
//!   避免中间张量；多维实部/虚部切分也可由步长视图代替拷贝；
//! - 在量化后端中，旋转不改变向量范数（正交性），可安全前置/后置量化。

use burn::module::Module;
use burn::tensor::Tensor;

/// RoPE 旋转位置编码算子。
///
/// 仅持有预计算的频率张量 `freq_cis`，**无可训练参数**，不参与优化器更新。
///
/// - `freq_cis`：`[1, 1, max_seq_len, head_dim / 2]`，第 3 维是位置索引、
///   第 4 维是 \(\theta_i\)（`freq_cis[m][i] = m·θ_i`，即角度而非复数）。
#[derive(Module, Debug)]
pub struct RoPE {
    pub freq_cis: Tensor<4>, // [1, 1, max_seq_len, head_dim / 2]
}

impl RoPE {
    /// 构造 RoPE 频率表。
    ///
    /// - `dim`：单头维度 `head_dim`（= `hidden_dim / num_heads`）；
    /// - `max_seq_len`：支持的最大序列长度（预计算上限）；
    /// - `theta`：基频（默认 10000.0）；
    /// - `device`：张量设备。
    pub fn new(dim: usize, max_seq_len: usize, theta: f32, device: &burn::tensor::Device) -> Self {
        // 1) 计算旋转频率 θ_i = theta ^ (-2i/d)
        //    实现：1 / theta^(i / d)，i 取 0, 2, 4, ...，得到 dim/2 个频率
        let mut inv_freq = Vec::with_capacity(dim / 2);
        for i in (0..dim).step_by(2) {
            let freq = 1.0 / theta.powf(i as f32 / dim as f32);
            inv_freq.push(freq);
        }

        // 频率向量 [dim / 2]
        let inv_freq = Tensor::<1>::from_data(inv_freq.as_slice(), device);

        // 2) 位置向量 [max_seq_len]：m = 0, 1, 2, ...
        let t = Tensor::<1>::from_data(
            (0..max_seq_len)
                .map(|x| x as f32)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        );

        // 3) 外积得到角度矩阵 [max_seq_len, dim / 2]：freqs[m][i] = m * θ_i
        let t = t.reshape([max_seq_len, 1]);
        let inv_freq = inv_freq.reshape([1, dim / 2]);
        let freqs = t.matmul(inv_freq);

        // 4) 扩展为 4 维 [1, 1, max_seq_len, dim / 2] 便于在 forward 中广播
        let freqs = freqs.reshape([1, 1, max_seq_len, dim / 2]);

        Self { freq_cis: freqs }
    }

    /// 应用 RoPE 旋转（从位置 0 开始，等价于 `forward_with_offset(x, 0)`）。
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.forward_with_offset(x, 0)
    }

    /// 应用 RoPE 旋转（带缓存偏移量，解码阶段使用）。
    ///
    /// 输入 `x: [Batch, Heads, SeqLen, HeadDim]`（要求 `D = 4`，
    /// 泛型保留是为了向后兼容调用处的统一签名）。
    pub fn forward_with_offset<const D: usize>(&self, x: Tensor<D>, offset: usize) -> Tensor<D> {
        // 形状推导
        let dims = x.dims();
        let seq_len = dims[D - 2];
        let head_dim = dims[D - 1];

        // 1) reshape: [.., Seq, HeadDim] -> [.., Seq, HeadDim/2, 2]
        let x_reshaped = x
            .clone()
            .reshape([dims[0], dims[1], seq_len, head_dim / 2, 2]);

        // 2) 切出相邻两半：x0（实部）/ x1（虚部），各 [.., Seq, HeadDim/2]
        let x0 = x_reshaped
            .clone()
            .slice([0..dims[0], 0..dims[1], 0..seq_len, 0..(head_dim / 2), 0..1])
            .squeeze_dim(4);
        let x1 = x_reshaped
            .slice([0..dims[0], 0..dims[1], 0..seq_len, 0..(head_dim / 2), 1..2])
            .squeeze_dim(4);

        // 3) 取当前序列真实位置对应的频率：slice [offset, offset+seq_len)
        let freqs =
            self.freq_cis
                .clone()
                .slice([0..1, 0..1, offset..offset + seq_len, 0..(head_dim / 2)]);

        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        // 4) 二维旋转：
        //    rx0 = x0·cos - x1·sin
        //    rx1 = x0·sin + x1·cos
        let rx0 = x0.clone() * cos.clone() - x1.clone() * sin.clone();
        let rx1 = x0 * sin + x1 * cos;

        // 5) 把 (rx0, rx1) 沿新轴拼回 [.., Seq, HeadDim/2, 2] 再还原原形状
        let rx0 = rx0.unsqueeze_dim(4);
        let rx1 = rx1.unsqueeze_dim(4);

        Tensor::<5>::cat(vec![rx0, rx1], 4).reshape(dims)
    }
}
