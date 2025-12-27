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
}

impl MetaIConfig {
    /// 1M 参数规模 (测试用)
    pub fn tiny() -> Self {
        Self {
            vocab_size: 16000,
            hidden_dim: 128,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 2, // GQA
            mlp_dim: 340,    // 约 8/3 * hidden_dim
            max_seq_len: 1024,
            dropout: 0.1,
            num_experts: 8,
            active_experts: 2,
        }
    }

    /// Local Device Optimized Config (e.g. M2/M3 Pro 16G)
    pub fn local_optimized() -> Self {
        Self {
            vocab_size: 16000,
            hidden_dim: 256, // 略微增加宽度
            num_layers: 6,   // 略微增加深度
            num_heads: 8,
            num_kv_heads: 4,
            mlp_dim: 680,
            max_seq_len: 512, // 关键：降低上下文长度以节省显存
            dropout: 0.1,
            num_experts: 4, // 专家数减半，适配 16G 内存
            active_experts: 2,
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
            num_experts: 8,    // 启用 MoE: 8 专家
            active_experts: 2, // Top-2 激活
        }
    }

    /// Llama 3 8B 参数规模 (当前 SOTA 此量级模型)
    /// 注意: 8x7B MoE 需要 40G+ 显存，因此在 16G 设备上最优选是 Llama 3 8B Dense
    pub fn llama_3_8b() -> Self {
        Self {
            vocab_size: 128256, // Llama 3 tokenizer
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,   // GQA
            mlp_dim: 14336,    // SwiGLU hidden dim
            max_seq_len: 8192, // 长上下文
            dropout: 0.0,
            num_experts: 1, // 保持 Dense 以适配 16G 内存
            active_experts: 1,
        }
    }
}
