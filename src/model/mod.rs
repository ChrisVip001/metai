use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

pub mod attention;
pub mod config;
pub mod mlp;
pub mod norm;
pub mod positional_encoding;

pub use attention::GroupedQueryAttention;
pub use config::MetaIConfig;
pub use mlp::MoE;
pub use norm::RMSNorm;
pub use positional_encoding::RoPE;

/// Transformer 块，包含注意力机制和 MoE MLP
/// 
/// 每个块包含：
/// - Grouped Query Attention (GQA) 用于高效注意力计算
/// - Mixture of Experts (MoE) MLP 用于条件计算
/// - RMSNorm 用于归一化
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub attention: GroupedQueryAttention<B>,
    pub mlp: MoE<B>,
    pub attn_norm: RMSNorm<B>,
    pub mlp_norm: RMSNorm<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &MetaIConfig, device: &B::Device) -> Self {
        Self {
            attention: GroupedQueryAttention::new(
                config.hidden_dim,
                config.num_heads,
                config.num_kv_heads,
                device,
            ),
            mlp: MoE::new(
                config.hidden_dim,
                config.mlp_dim,
                config.num_experts,
                config.active_experts,
                device,
            ),
            attn_norm: RMSNorm::new(config.hidden_dim, 1e-5, device),
            mlp_norm: RMSNorm::new(config.hidden_dim, 1e-5, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RoPE<B>,
        mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
        cache: Option<&mut crate::infer::cache::KVCache<B>>,
    ) -> Tensor<B, 3> {
        let h = x.clone()
            + self
                .attention
                .forward(self.attn_norm.forward(x), rope, mask.clone(), cache);
        h.clone() + self.mlp.forward(self.mlp_norm.forward(h))
    }
}

/// MetaI 语言模型主结构
/// 
/// 实现了基于 Transformer 的 LLM，支持：
/// - 训练和推理模式
/// - KV Cache 加速推理
/// - Padding Mask 处理变长序列
/// - INT4 量化支持
#[derive(Module, Debug)]
pub struct MetaIModel<B: Backend> {
    pub embedding: Embedding<B>,
    pub blocks: Vec<TransformerBlock<B>>,
    pub norm: RMSNorm<B>,
    pub output: Linear<B>,
    pub rope: RoPE<B>,
    pub pad_id: u32,
}

impl<B: Backend> MetaIModel<B> {
    pub fn new(config: &MetaIConfig, pad_id: u32, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(config.vocab_size, config.hidden_dim).init(device);
        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(config, device));
        }
        let norm = RMSNorm::new(config.hidden_dim, 1e-5, device);
        let output = LinearConfig::new(config.hidden_dim, config.vocab_size)
            .with_bias(false)
            .init(device);
        let rope = RoPE::new(
            config.hidden_dim / config.num_heads,
            config.max_seq_len,
            10000.0,
            device,
        );

        Self {
            embedding,
            blocks,
            norm,
            output,
            rope,
            pad_id,
        }
    }

    /// 模型前向传播
    /// 
    /// # 参数
    /// - `tokens`: 输入 token IDs，形状为 [batch_size, seq_len]
    /// - `_mask`: 可选的注意力掩码（当前未使用，使用自动生成的 padding mask）
    /// - `cache`: 可选的 KV Cache，用于加速推理
    /// 
    /// # 返回
    /// 输出 logits，形状为 [batch_size, seq_len, vocab_size]
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        _mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
        cache: Option<&mut crate::infer::cache::ModelCache<B>>,
    ) -> Tensor<B, 3> {
        // 自动生成 Padding Mask
        // 如果输入 Token 等于 pad_id，则视为 Padding (True)
        // equal_elem 返回 Bool Tensor
        let padding_mask = tokens.clone().equal_elem(self.pad_id as i32);

        let mut h = self.embedding.forward(tokens);

        if let Some(cache) = cache {
            // Inference with Cache
            for (block, layer_cache) in self.blocks.iter().zip(cache.layers.iter_mut()) {
                h = block.forward(h, &self.rope, Some(padding_mask.clone()), Some(layer_cache));
            }
        } else {
            // Training or No-Cache
            for block in &self.blocks {
                h = block.forward(h, &self.rope, Some(padding_mask.clone()), None);
            }
        }

        h = self.norm.forward(h);
        self.output.forward(h)
    }

    /// 执行 INT4 权重转换 (PTQ - Weight Only)
    /// 
    /// 使用块级量化方案，将模型权重压缩为 4-bit 整数。
    /// 这可以大幅减少模型大小和推理时的内存占用。
    /// 
    /// # 注意
    /// 量化后的模型精度可能会有所下降，建议在量化后进行校准。
    pub fn quantize_int4(self, _device: &B::Device) -> Self {
        use burn::module::Quantizer;
        use burn::tensor::quantization::{
            BlockSize, Calibration, QuantLevel, QuantScheme, QuantValue,
        };

        // 定义 INT4 分块量化方案: Q4S (4-bit Signed), 32个元素一个 Block
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q4S)
            .with_level(QuantLevel::Block(BlockSize::new([32])));

        let mut quantizer = Quantizer {
            calibration: Calibration::MinMax,
            scheme,
        };

        // 仅对权重进行量化转换
        self.quantize_weights(&mut quantizer)
    }
}
