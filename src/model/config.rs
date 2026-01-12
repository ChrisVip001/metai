use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaIConfig {
    pub vocab_size: usize,         // 词表大小
    pub hidden_dim: usize,         // 隐藏层维度
    pub num_layers: usize,         // 层数
    pub num_heads: usize,          // 头数
    pub num_kv_heads: usize,       // KV Cache 头数
    pub mlp_dim: usize,            // MLP 层维度
    pub max_seq_len: usize,        // 最大序列长度
    pub dropout: f64,              // Dropout 概率
    pub num_experts: usize,        // 专家数量
    pub active_experts: usize,     // 激活的专家数量
    pub num_shared_experts: usize, // 共享的专家数量
}

impl MetaIConfig {
    /// 5M 参数规模 (Nano 版)
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
        }
    }

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
}
