//! # SwiGLU —— 门控前馈激活算子
//!
//! 前馈网络（FFN）是 Transformer 中逐 token 的非线性变换。SwiGLU 在
//! LLM 中取代了早期 `ReLU(W₁x)W₂` 的「两投影」结构，被 LLaMA、Mistral、
//! DeepSeek、Qwen 等主流模型采用。
//!
//! ## 数学定义
//!
//! 本项目（与 LLaMA / DeepSeek 一致）使用「无 bias、三权重」的 SwiGLU：
//!
//! \[
//! \text{SwiGLU}(x) = W_2\Big(\mathrm{SiLU}(W_1 x) \odot (W_3 x)\Big)
//! \]
//!
//! - `w1`（门控投影）：`x → gate`，过 SiLU 激活后**充当开关**；
//! - `w3`（数据投影）：`x → value`，承载被门控的真实信息；
//! - `w2`（输出投影）：把组合结果投影回 `hidden_dim`。
//!
//! 逐元素运算示意（`h` 为中间维 `mlp_dim`）：
//!
//! ```text
//! x [..,H]
//!   ├─ w1 → [..,h] → SiLU ─────────┐
//!   │                               ├─ ⊙（逐元素乘）→ [..,h]
//!   └─ w3 → [..,h] ─────────────────┘
//!                                     └─ w2 → [..,H]
//! ```
//!
//! ## 为什么是 SiLU（Sigmoid Linear Unit）
//!
//! \[
//! \mathrm{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1+e^{-z}}
//! \]
//!
//! SiLU 是平滑、单调的 S 形函数：
//!
//! - **处处可导**（对比 ReLU 在 0 点不可导），梯度更稳定；
//! - 对负输入产生**非零的小梯度**（对比 ReLU 硬截断为 0，避免了「死神经元」）；
//! - 在约 `-1.28` 处有平滑下界（非饱和），经验上比 GELU 更稳。
//!
//! 门控 \(g = \mathrm{SiLU}(W_1 x)\) 取值范围为负到正，可近似看作「软开关」：
//! 值为负时抑制对应数据维度，为正时放大——这正是门控（gating）的意义：
//! **模型为每个输入自适应地选择要放行哪些特征**。
//!
//! ## 参数量与 FLOPs
//!
//! 三权重结构共有 `3 × hidden × mlp_dim` 参数（无 bias），约为传统
//! `ReLU` 双投影 FFN（`2 × hidden × mlp_dim`）的 1.5 倍。为公平比较，
//! 模型设计时会把 `mlp_dim` 调小约 1/3（`config.mlp_dim = 2.5~3 × hidden_dim`
//! 即可），使总参数量与经典 FFN 相当，同时表达能力更强。
//!
//! ## 独立优化方向（本算子即替换点）
//!
//! - SiLU 与逐元素乘可融合为一个 kernel（gate 计算完立刻与 value 相乘，
//!   避免多一次全局内存写读）；
//! - `w1/w3` 两投影共享输入 `x`，可把两矩阵拼接为一个大权重做**一次 GEMM**
//!   （fused QKV 风格），减少 kernel 启动开销；
//! - 权重转置布局（`[out, in]` 或 `[in, out]`）按后端 GEMM 偏好切换。

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::Tensor;

/// SwiGLU 门控前馈算子。
///
/// 字段说明：
/// - `w1`：门控投影 \(H \to h\)；
/// - `w2`：输出投影 \(h \to H\)；
/// - `w3`：数据（value）投影 \(H \to h\)。
///
/// 三个投影均无 bias（现代 LLM 去 bias 约定）。
#[derive(Module, Debug)]
pub struct SwiGLU {
    pub w1: Linear, // 门控层投影: x -> silu(gate)
    pub w2: Linear, // 输出层投影: 结果 -> hidden_dim
    pub w3: Linear, // 数据层投影: x -> value
}

impl SwiGLU {
    /// 构造 SwiGLU（`hidden_dim` → `mlp_dim` → `hidden_dim`）。
    pub fn new(hidden_dim: usize, mlp_dim: usize, device: &burn::tensor::Device) -> Self {
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

    /// 前向传播：`w2( silu(w1(x)) ⊙ w3(x) )`。
    ///
    /// 泛型维度 `D` 支持任意形状（`[B, S, H]`、展平的 `[T, H]`、
    /// 以及 MoE 中按专家 Gather 出的子批 `[N, H]`）。
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        // 门控分支：silu(w1(x))
        let gate = silu(self.w1.forward(x.clone()));
        // 数据分支：w3(x)
        let data = self.w3.forward(x);
        // 门控放行：逐元素相乘
        let combined = gate * data;
        // 输出投影回 hidden_dim
        self.w2.forward(combined)
    }
}
