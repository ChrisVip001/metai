use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaIConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub mlp_dim: usize,
    pub max_seq_len: usize,
    pub dropout: f64,
    pub num_experts: usize,
    pub active_experts: usize,
    pub num_shared_experts: usize,
}

impl MetaIConfig {
    /// 125M 参数规模 (MoE 增强版)
    /// 虽然总参数量增加，但推理/训练计算量保持在 125M Dense 级别，效果更优
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
        }
    }
    /// DeepSeek-V3 规模 (671B MoE)
    /// 这里的配置模拟了 671B 总参数，256 专家，Top-8 激活的架构。
    /// 注意：在 2026 年，这需要多机多卡 FSDP 或 Pipeline Parallelism 运行。
    pub fn deepseek_v3_671b() -> Self {
        Self {
            vocab_size: 129280,
            hidden_dim: 7168,
            num_layers: 61,
            num_heads: 128,
            num_kv_heads: 1, // MLA (Multi-Head Latent Attention) 简化模拟
            mlp_dim: 18432,
            max_seq_len: 32768,
            dropout: 0.0,
            num_experts: 256,
            active_experts: 8,
            num_shared_experts: 1,
        }
    }
}
