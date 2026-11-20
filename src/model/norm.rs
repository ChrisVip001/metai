//! # 归一化算子族（Normalization Family）
//!
//! 归一化是 Transformer 稳定训练的基石。本文件按算子级可替换的思路，
//! 把一个归一化族拆成 **三个可独立替换的具体算子** + **一个分发容器**：
//!
//! | 变体 | 公式 | 权重 | 里程碑 |
//! |---|---|---|---|
//! | [`RMSNorm`]（默认） | \(x / \sqrt{\mathrm{mean}(x^2)+\varepsilon} \cdot \gamma\) | γ | LLaMA 系（2019/2023） |
//! | [`LayerNorm`] | \((x-\mu)/\sqrt{\sigma^2+\varepsilon} \cdot \gamma + \beta\) | γ, β | BERT/GPT-2（2016） |
//! | [`DyT`] | \(\tanh(\alpha \cdot x)\) | α | CVPR 2025 / Meta BLT |
//!
//! [`Norm`] 是组合层的装配点：配置选择 [`NormVariant`]，`TransformerBlock`
//! 无需感知具体实现。三者数学定义与形状约束见各算子 doc。

use burn::module::{Module, Param};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

// ================================================================
// RMSNorm —— 均方根归一化（默认变体）
// ================================================================

/// RMSNorm 归一化算子（含可学习缩放参数 `gamma`）。
///
/// 结构字段：
/// - `gamma`：逐通道缩放权重，形状 `[dim]`，初始化全 1；
/// - `epsilon`：数值稳定常数（RMS 公式中的 \(\varepsilon\)）。
#[derive(Module, Debug)]
pub struct RMSNorm {
    pub gamma: Param<Tensor<1>>,
    pub epsilon: f64,
}

impl RMSNorm {
    /// 构造 RMSNorm。
    ///
    /// - `dim`：归一化维度大小（通常等于 `hidden_dim`）；
    /// - `epsilon`：RMS 分母中的小常数；
    /// - `device`：权重所在设备。
    pub fn new(dim: usize, epsilon: f64, device: &burn::tensor::Device) -> Self {
        let gamma = Param::from_tensor(Tensor::ones([dim], device));
        Self { gamma, epsilon }
    }

    /// 前向归一化：
    ///
    /// ```text
    /// y = x / sqrt(mean(x², last_dim) + eps) * gamma
    /// ```
    ///
    /// 泛型 `D` 使本算子可用于 `[B, S, H]`（块内 norm）以及任意维输入。
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let dim = x.dims()[D - 1];
        let eps = self.epsilon;

        // 1) 计算 RMS：mean(x²) 后开方（mean_dim 保留单位尾维便于广播）
        let norm = (x.clone().powf_scalar(2.0).mean_dim(D - 1) + eps).sqrt();

        // 2) 逐元素缩放归一化
        let x = x / norm;

        // 3) 应用可学习 gamma，构造广播形状 [1, ..., 1, dim]
        let gamma = self.gamma.val();
        let mut shape = [1; D];
        shape[D - 1] = dim;

        x * gamma.reshape(shape)
    }
}

// ================================================================
// LayerNorm —— 层归一化（含均值中心化，BERT/GPT-2 时代标准）
// ================================================================

/// LayerNorm 归一化算子（含 γ/β 两个可学习参数）。
///
/// 相比 RMSNorm，先做均值中心化再按方差缩放：
///
/// ```text
/// μ  = mean(x)                     （沿最后一维）
/// σ² = mean((x - μ)²)
/// y  = (x - μ) / sqrt(σ² + eps) * γ + β
/// ```
#[derive(Module, Debug)]
pub struct LayerNorm {
    pub gamma: Param<Tensor<1>>,
    pub beta: Param<Tensor<1>>,
    pub epsilon: f64,
}

impl LayerNorm {
    /// 构造 LayerNorm。`gamma` 初始为全 1，`beta` 初始为全 0。
    pub fn new(dim: usize, epsilon: f64, device: &burn::tensor::Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([dim], device)),
            beta: Param::from_tensor(Tensor::zeros([dim], device)),
            epsilon,
        }
    }

    /// 前向层归一化（均值中心化 + 方差缩放 + 仿射）。
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let dim = x.dims()[D - 1];
        let eps = self.epsilon;

        // 1) 均值中心化（mean_dim 保留单位尾维便于广播）
        let mean = x.clone().mean_dim(D - 1);
        let centered = x.clone() - mean;

        // 2) 方差 + 缩放（biased variance：除以 N，LN 惯例）
        let variance = centered.clone().powf_scalar(2.0).mean_dim(D - 1);
        let normed = centered / (variance + eps).sqrt();

        // 3) 仿射：* γ + β，广播形状 [1, ..., 1, dim]
        let mut shape = [1; D];
        shape[D - 1] = dim;
        let gamma = self.gamma.val().reshape(shape);
        let beta = self.beta.val().reshape(shape);

        normed * gamma + beta
    }
}

// ================================================================
// DyT —— Dynamic Tanh（无归一化 Transformer，CVPR 2025 / Meta BLT）
// ================================================================

/// DyT（Dynamic Tanh）逐元素算子：\(y = \tanh(\alpha \cdot x)\)。
///
/// 观察发现 Transformer 中归一化层的输入-输出映射接近 tanh 形状，因此可以
/// 用**一个逐元素 tanh + 单个可学习标量 α** 直接替换整个归一化层，从而：
///
/// - 消除跨 token 的 reduction（mean/var），更利于 kernel 融合与长序列；
/// - 保留 S 形软饱和特性，训练效果与 LayerNorm/RMSNorm 相当或更优；
/// - `alpha` 初始值参考论文取 `0.5`，随训练自适应。
#[derive(Module, Debug)]
pub struct DyT {
    pub alpha: Param<Tensor<1>>,
}

impl DyT {
    /// 构造 DyT。`alpha` 为形状 `[1]` 的可学习标量，初始化 `0.5`。
    pub fn new(device: &burn::tensor::Device) -> Self {
        Self {
            alpha: Param::from_tensor(Tensor::from_data([0.5f32], device)),
        }
    }

    /// 前向：`y = tanh(α·x)`。`alpha` 广播到任意维度输入。
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let alpha = self.alpha.val().reshape([1; D]);
        (x * alpha).tanh()
    }
}

// ================================================================
// 变体枚举 + 分发容器（组合层装配点）
// ================================================================

/// 归一化族变体标识（配置序列化用，默认 `RmsNorm` 保持现状行为）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NormVariant {
    /// RMSNorm（现代 LLM 默认，本仓库默认）
    #[default]
    RmsNorm,
    /// LayerNorm（BERT/GPT-2 传统归一化）
    LayerNorm,
    /// DyT（CVPR 2025，无归一化 Transformer）
    DyT,
}

/// 归一化算子分发容器。
///
/// 这是「组合算子可替换」的第一级装配点：`TransformerBlock` 只依赖本
/// 容器，具体归一化变体由配置决定，替换变体不影响块与模型层代码。
#[derive(Module, Debug)]
pub enum Norm {
    Rms(RMSNorm),
    Layer(LayerNorm),
    DyT(DyT),
}

impl Norm {
    /// 按变体构造归一化算子。
    ///
    /// - `dim`：归一化维度（DyT 不需要，但为统一接口保留）；
    /// - `epsilon`：数值稳定常数；
    /// - `variant`：目标变体；
    /// - `device`：权重设备。
    pub fn new(
        dim: usize,
        epsilon: f64,
        variant: NormVariant,
        device: &burn::tensor::Device,
    ) -> Self {
        match variant {
            NormVariant::RmsNorm => Norm::Rms(RMSNorm::new(dim, epsilon, device)),
            NormVariant::LayerNorm => Norm::Layer(LayerNorm::new(dim, epsilon, device)),
            NormVariant::DyT => Norm::DyT(DyT::new(device)),
        }
    }

    /// 统一前向入口：按内部变体转发（各变体实现见上）。
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        match self {
            Norm::Rms(rms) => rms.forward(x),
            Norm::Layer(ln) => ln.forward(x),
            Norm::DyT(dyt) => dyt.forward(x),
        }
    }
}
