use crate::model::norm::NormVariant;
use crate::model::router::RouterVariant;
use crate::model::sparse::SparseAttentionConfig;
use serde::{Deserialize, Serialize};

/// MLA（Multi-head Latent Attention，DeepSeek-V2 结构族）压缩参数。
///
/// 核心思想：把每层 KV 先投影到低秩 latent \(c_t = W_{DKV} h_t\) 再使用，
/// cache 只存 \(c\)（+ 解耦 RoPE 键），注意力打分在 latent 空间进行，
/// 每 token 缓存开销从「每头全量 KV」降为 \(kv\_lora\_rank + rope\) 一列
/// （与头数解耦）。结构差异见 [`crate::model::attention`] / [`AttentionVariant`]。
///
/// DeepSeek-V2 生产值参考：`kv_lora_rank = 512`，`qk_rope_head_dim = 64`
/// （头内分 nope / rope 两段，本仓库以「吸收式 + 共享 rope」等价落地，
/// 文档见 `docs/INDUSTRIAL.zh-CN.md`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlaConfig {
    /// 压缩 latent 维度（cache 中每 token 的主体存储宽度）。
    pub kv_lora_rank: usize,
    /// 解耦 RoPE 维度（单通道共享，不随头数增长）。
    pub qk_rope_head_dim: usize,
}

impl MlaConfig {
    /// 构造 MLA 配置（参考 DeepSeek-V2 比例：rope ≈ latent 的 1/8）。
    pub fn new(kv_lora_rank: usize, qk_rope_head_dim: usize) -> Self {
        Self {
            kv_lora_rank,
            qk_rope_head_dim,
        }
    }
}

/// 注意力结构变体（配置序列化用，默认 `Gqa` 保持现状行为）。
///
/// - `Gqa`：分组查询注意力（默认，见 [`crate::model::attention::GroupedQueryAttention`]）；
/// - `Mla(MlaConfig)`：低秩 latent 注意力（DeepSeek-V2 族，见 [`crate::model::attention::MultiLatentAttention`]）。
///
/// `num_kv_heads` 只对 `Gqa` 生效；`Mla` 下 KV 头数被 latent 通道取代。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AttentionVariant {
    #[default]
    Gqa,
    Mla(MlaConfig),
}

/// 模型架构配置：除尺寸外，还可在「算子族」内选择实现变体。
///
/// 变体字段均带 `#[serde(default)]`，缺省时回退到默认变体，
/// 保证旧配置/checkpoint 结构不因新增变体字段而失效。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaIConfig {
    pub vocab_size: usize,         // 词表大小
    pub hidden_dim: usize,         // 隐藏层维度
    pub num_layers: usize,         // 层数
    pub num_heads: usize,          // 头数
    pub num_kv_heads: usize, // KV Cache 头数（= n_heads 时退化为 MHA，= 1 时退化为 MQA；仅 GQA 生效）
    pub mlp_dim: usize,      // MLP 层维度
    pub max_seq_len: usize,  // 最大序列长度
    pub dropout: f64,        // Dropout 概率
    pub num_experts: usize,  // 专家数量
    pub active_experts: usize, // 激活的专家数量
    pub num_shared_experts: usize, // 共享的专家数量
    /// 归一化族变体（默认 `RmsNorm`）。
    #[serde(default)]
    pub norm: NormVariant,
    /// MoE 路由变体（默认 `SoftmaxTopK`）。
    #[serde(default)]
    pub router: RouterVariant,
    /// KV 侧稀疏注意力（NSA/MoBA/DSA 的「位置打分 + 选择」算子族，
    /// 见 [`crate::model::sparse`]）；`None` = 传统 dense 注意力（默认）。
    /// 注意：当前仅与 `AttentionVariant::Gqa` 组合（MLA×稀疏为后续任务）。
    #[serde(default)]
    pub sparse: Option<SparseAttentionConfig>,
    /// 注意力结构变体（默认 `Gqa`）。
    #[serde(default)]
    pub attn: AttentionVariant,
}

impl MetaIConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.vocab_size >= 4
                && self.max_seq_len >= 2
                && self.num_layers > 0
                && self.mlp_dim > 0,
            "invalid model sizes"
        );
        anyhow::ensure!(
            self.num_heads > 0
                && self.num_kv_heads > 0
                && self.hidden_dim > 0
                && self.hidden_dim.is_multiple_of(self.num_heads)
                && self.num_heads.is_multiple_of(self.num_kv_heads)
                && (self.hidden_dim / self.num_heads).is_multiple_of(2),
            "invalid head dimensions"
        );
        anyhow::ensure!(
            self.active_experts > 0 && self.active_experts <= self.num_experts,
            "invalid expert count"
        );
        anyhow::ensure!(self.dropout == 0., "dropout is not implemented; use 0");
        if let AttentionVariant::Mla(m) = self.attn {
            anyhow::ensure!(
                self.sparse.is_none()
                    && m.kv_lora_rank > 0
                    && m.qk_rope_head_dim > 0
                    && m.qk_rope_head_dim.is_multiple_of(2),
                "invalid MLA configuration"
            );
        }
        Ok(())
    }

    /// CPU 教学配置，保留完整注意力、路由和反向传播。
    pub fn tiny() -> Self {
        Self {
            vocab_size: 320,
            hidden_dim: 16,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            mlp_dim: 32,
            max_seq_len: 96,
            num_experts: 2,
            active_experts: 2,
            ..Self::micro()
        }
    }

    /// 单机 Micro 预设。
    /// 极其适合笔记本快速验证逻辑，训练速度飞快
    pub fn micro() -> Self {
        Self {
            vocab_size: 16000,
            hidden_dim: 128,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 2,
            mlp_dim: 384,     // 3 * hidden_dim
            max_seq_len: 256, // 缩短序列
            dropout: 0.0,
            num_experts: 4,    // 4 专家
            active_experts: 1, // Top-1 激活
            num_shared_experts: 0,
            norm: NormVariant::RmsNorm,
            router: RouterVariant::SoftmaxTopK,
            sparse: None,
            attn: AttentionVariant::Gqa,
        }
    }

    /// Single-node 125M-class GQA/SwiGLU reference with the repository's 16K tokenizer.
    pub fn prototype_125m() -> Self {
        Self {
            vocab_size: 16000,
            num_layers: 16,
            num_experts: 1,
            active_experts: 1,
            num_shared_experts: 0,
            ..Self::small()
        }
    }

    /// 更宽、更深的多专家 Small 预设。
    pub fn small() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 4,
            mlp_dim: 2048,
            max_seq_len: 2048,
            dropout: 0.0,
            num_experts: 8,        // 启用 MoE: 8 专家
            active_experts: 2,     // Top-2 激活
            num_shared_experts: 2, // 默认开启 2 个共享专家以测试
            norm: NormVariant::RmsNorm,
            router: RouterVariant::SoftmaxTopK,
            sparse: None,
            attn: AttentionVariant::Gqa,
        }
    }

    /// 685B MoE 规模 (685B MoE)
    /// - 671B Base + 14B MTP (Multi-Token Prediction) = 685B
    /// - 128K Context Window
    /// - 256 Experts, Top-8 Active
    pub fn deepseek_v3_2_685b() -> Self {
        Self {
            vocab_size: 129280,
            hidden_dim: 7168,
            num_layers: 61,
            num_heads: 128,
            num_kv_heads: 1, // MLA 潜空间压缩
            mlp_dim: 18432,
            max_seq_len: 131072, // 128K + 辅助空间
            dropout: 0.0,
            num_experts: 256,
            active_experts: 8,
            num_shared_experts: 1,
            norm: NormVariant::RmsNorm,
            // V3.2 采用 sigmoid 无 softmax 路由 + bias 均衡（DeepSeek-V3 风格）
            router: RouterVariant::SigmoidTopK,
            // V3.2 标配 DSA 稀疏注意力；此处默认仍 dense，
            // 显式设置后走本仓库的稀疏 mask 位（见 docs/INDUSTRIAL.zh-CN.md）。
            // 注意稀疏位当前仅与 Gqa 组合（见 attn 字段注释）。
            sparse: None,
            // MLA：KV 低秩 latent 压缩 + 解耦 RoPE（DeepSeek-V2 尺寸；
            // n_kv_heads 在 MLA 下被 latent 通道取代，不再直接使用）
            attn: AttentionVariant::Mla(MlaConfig::new(512, 64)),
        }
    }
}
