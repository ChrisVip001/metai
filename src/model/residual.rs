//! # 残差连接算子（Residual Connection Operator）
//!
//! ## 数学定义
//!
//! 残差连接是 Transformer 块的基本结构单元：
//!
//! \[
//! y = x + \mathcal{F}(x)
//! \]
//!
//! 其中 \(\mathcal{F}\) 可以是注意力子层或前馈（MoE）子层。本算子只负责
//! 「输入 \(x\) 与子层输出 \(\mathcal{F}(x)\) 的逐元素相加」，把这条路独立出来，
//! 以便：
//!
//! 1. 统一控制残差路径（如切换为 *ReZero* / *LayerScale* 等加权残差变体）；
//! 2. 在分布式/流水线场景对残差点做通信与同步；
//! 3. 单独进行 profiling 与数值检查。
//!
//! ## Pre-Norm 结构说明
//!
//! 本项目（与 LLaMA / DeepSeek 一致）采用 **Pre-Norm** 布局：
//!
//! \[
//! \text{block}(x) = x + \mathcal{F}\big(\text{Norm}(x)\big)
//! \]
//!
//! 归一化放在子层**之前**。相比 Post-Norm（\(\text{Norm}(x + \mathcal{F}(x))\)），
//! Pre-Norm 有两点重要性质：
//!
//! - **梯度捷径（gradient highway）**：恒等路径不经过任何非线性与归一化，
//!   使反向传播时 \(\partial L / \partial x\) 可以直接从高层流向低层，
//!   缓解深层网络梯度消失，因此可以训练到极深（数百层）；
//! - 代价是残差路径与分支路径**量纲不一致**（x 已稳定，而分支输出无界），
//!   实践中靠归一化与合适的初始化弥补，通常不需要额外的 LayerScale。
//!
//! ## 可升级方向
//!
//! - 缩放残差：\(y = \alpha \cdot x + \mathcal{F}(x)\)（*ReZero*：\(\alpha\) 初值 0，
//!   训练初期等价于浅层网络，稳定不爆炸；*LayerScale*：逐通道 \(\alpha\)）；
//! - 在残差路径上叠加 Dropout / Stochastic Depth（随机丢弃整层）；
//! - 训练时可在此处插入梯度裁剪 / activation checkpointing 钩子。
//!
//! ## 形状约束
//!
//! \[
//! x,\ \mathcal{F}(x) \in \mathbb{R}^{B \times S \times H} \Rightarrow
//! y \in \mathbb{R}^{B \times S \times H}
//! \]
//!
//! 本算子为**无参数**算子（不含任何可训练权重，不进入 checkpoint），
//! 泛化到任意维度张量使用。

use burn::tensor::Tensor;

/// 残差连接算子：`forward(x, branch) = x + branch`。
///
/// 结构体本身无字段；若未来需要 *ReZero* / *LayerScale* 等带权残差，
/// 只需在此结构中增加 `Param`/标量字段并在 `forward` 中乘上缩放因子，
/// 调用方（`TransformerBlock`）无需改动。
#[derive(Debug, Clone, Copy, Default)]
pub struct Residual;

impl Residual {
    /// 计算 `x + branch`（残差捷径 + 子层输出）。
    ///
    /// - `x`：残差路径输入（块的原始输入，**未**经过归一化）；
    /// - `branch`：子层（Norm + Attention / Norm + MoE）的输出。
    pub fn forward<const D: usize>(x: Tensor<D>, branch: Tensor<D>) -> Tensor<D> {
        x + branch
    }

    /// [`Self::forward`] 的别名，语义更贴近「把分支结果加回主流」。
    pub fn add<const D: usize>(x: Tensor<D>, branch: Tensor<D>) -> Tensor<D> {
        Self::forward(x, branch)
    }
}
