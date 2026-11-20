//! # MetaI 模型库：算子化 Transformer 架构
//!
//! 本模块把 Transformer 的计算链路显式拆分为一组**可独立优化升级的算子**。
//! 每个算子占用一个源文件，并附带完整的原理文档（数学定义、形状流转、
//! 复杂度与优化方向）。组合算子只负责「编排」，不内联其他算子的数学实现。
//!
//! ## 算子地图
//!
//! | 文件 | 算子族 | 默认变体 | 已具变体 | 是否有权重 |
//! |---|---|---|---|---|
//! | [`norm`] | 归一化 | `RMSNorm` | `LayerNorm` / `DyT`（经 `NormVariant`） | 是 |
//! | [`positional_encoding`] | 位置编码 | `RoPE` | 外推预留 | 否（频率表） |
//! | [`heads`] | 多头布局 | GQA | MHA / MQA（`num_kv_heads` 特例） | 否 |
//! | [`sdpa`] | 注意力核心 | naive SDPA | mask 组合 | 否 |
//! | [`sparse`] | KV 稀疏（语义位） | dense | 块 Top-k + 窗口 mask | 否 |
//! | [`sparse_exec`] | KV 稀疏（解码执行位） | dense | 紧凑 gather（物理省算） | 否 |
//! | [`attention`] | 注意力组合 | `GroupedQueryAttention` | 投影→拆头→旋转→SDPA→合并 | 是（w_q/k/v/o） |
//! | [`ffn`] | 前馈 | `SwiGLU` | — | 是（w1/w2/w3） |
//! | [`router`] | MoE 路由 | softmax Top-K | `SigmoidTopK`（DeepSeek-V3 风格） | 否（gate 在 MoE） |
//! | [`moe`] | 稀疏专家（组合） | 路由+共享专家 | — | 是（gate + 专家） |
//! | [`residual`] | 残差捷径 | 恒等 add | — | 否 |
//!
//! 替换变体的两种装配方式：
//! 1. **归一化/路由**：`config.norm` / `config.router` 选择变体，
//!    组合层经 [`norm::Norm`] 分发容器 / `router::route_with` 装配，无需改块；
//! 2. **注意力等结构级变体**：直接换 `attention.rs` 实现（如 MLA），
//!    块内仅改构造处。
//!
//! ## Transformer 块级算子流水线（[`TransformerBlock`]）
//!
//! ```text
//! x ──► [Norm attn_norm] ──► [GQA 注意力] ──► (+) 残差 ──► h
//! h ──► [Norm mlp_norm] ──► [MoE 稀疏前馈] ──► (+) 残差 ──► out
//! ```
//! （`Norm` 为归一化族分发容器，默认变体 = RMSNorm，见 [`norm`]）
//!
//! 对某一算子做内核级优化（FlashAttention、fused QKV、稀疏 MoE kernel 等）
//! 时，只需替换对应算子文件/实现，无需改动块与模型层。

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::{Int, Tensor};

pub mod attention;
pub mod config;
pub mod ffn;
pub mod heads;
pub mod mla;
pub mod moe;
pub mod norm;
pub mod positional_encoding;
pub mod residual;
pub mod router;
pub mod sdpa;
pub mod sparse;
pub mod sparse_exec;

pub use attention::{Attention, GroupedQueryAttention};
pub use config::{AttentionVariant, MetaIConfig, MlaConfig};
pub use ffn::SwiGLU;
pub use heads::{merge_heads, repeat_kv, split_heads};
pub use mla::MultiLatentAttention;
pub use moe::MoE;
pub use norm::{DyT, LayerNorm, Norm, NormVariant, RMSNorm};
pub use positional_encoding::RoPE;
pub use residual::Residual;
pub use router::{route, route_with, RouterVariant};
pub use sdpa::{build_causal_mask, combine_masks, scaled_dot_product_attention};
pub use sparse::{sparse_attention_mask, SparseAttentionConfig};

/// Transformer 块（组合算子）。
///
/// 由两个「归一化 + 子算子 + 残差」子层组成（Pre-Norm 布局）：
///
/// ```text
/// attn 子层：x + GQA(attn_norm(x))          ← 捕获 token 间关系
/// ffn  子层：h + MoE(mlp_norm(h))           ← 逐 token 非线性变换
/// ```
#[derive(Module, Debug)]
pub struct TransformerBlock {
    /// 注意力结构变体容器（GQA 或 MLA，依 `config.attn` 装配）。
    pub attention: Attention,
    pub mlp: MoE,
    /// 注意力子层归一化（归一化族分发容器，默认 RMSNorm）。
    pub attn_norm: Norm,
    /// 前馈子层归一化（同上）。
    pub mlp_norm: Norm,
}

impl TransformerBlock {
    /// 构造 Transformer 块：注意力（`config.attn` 选 GQA / MLA）+ MoE 前馈
    /// + 两个归一化。
    ///
    /// 归一化/路由的具体变体取自 `config.norm` / `config.router`；
    /// KV 侧稀疏注意力开关取自 `config.sparse`（当前仅 GQA 支持）；
    /// 替换注意力结构不影响块的编排逻辑。
    pub fn new(config: &MetaIConfig, device: &burn::tensor::Device) -> Self {
        let mut mlp = MoE::new(
            config.hidden_dim,
            config.mlp_dim,
            config.num_experts,
            config.active_experts,
            config.num_shared_experts,
            device,
        );
        mlp.set_router(config.router);

        let attention = match config.attn {
            AttentionVariant::Gqa => {
                let mut gqa = GroupedQueryAttention::new(
                    config.hidden_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    device,
                );
                gqa.set_sparse(config.sparse.clone());
                Attention::Gqa(gqa)
            }
            AttentionVariant::Mla(cfg) => {
                // MLA × KV 稀疏组合（对应 DeepSeek-V3.2 路线）为后续任务。
                assert!(
                    config.sparse.is_none(),
                    "MLA 与 config.sparse（KV 稀疏）组合尚未实现"
                );
                Attention::Mla(MultiLatentAttention::new(
                    config.hidden_dim,
                    config.num_heads,
                    cfg,
                    device,
                ))
            }
        };

        Self {
            attention,
            mlp,
            attn_norm: Norm::new(config.hidden_dim, 1e-5, config.norm, device),
            mlp_norm: Norm::new(config.hidden_dim, 1e-5, config.norm, device),
        }
    }

    /// 块前向：算子流水线编排（各算子原理见其模块文档）。
    ///
    /// - `x`：`[Batch, Seq, Hidden]`；
    /// - `rope`：共享 RoPE（含预计算频率）；
    /// - `mask`：padding mask `[Batch, Seq]`；
    /// - `cache`：可选 KV Cache（推理解码）。
    pub fn forward(
        &self,
        x: Tensor<3>,
        rope: &RoPE,
        mask: Option<Tensor<2, burn::tensor::Bool>>,
        cache: Option<&mut crate::infer::cache::KVCache>,
    ) -> Tensor<3> {
        // ---- 算子子层 1：注意力（RMSNorm → GQA → 残差） ----
        let residual = x.clone();
        let normed = self.attn_norm.forward(x);
        let attended = self.attention.forward(normed, rope, mask, cache);
        let h = Residual::add(residual, attended);

        // ---- 算子子层 2：前馈（RMSNorm → MoE → 残差） ----
        let residual = h.clone();
        let normed = self.mlp_norm.forward(h);
        let output = self.mlp.forward(normed);
        Residual::add(residual, output)
    }
}

/// MetaI 语言模型主结构（组合算子）。
///
/// `Token → Embedding → N×TransformerBlock → RMSNorm → Linear(词表)`。
#[derive(Module, Debug)]
pub struct MetaIModel {
    pub embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    /// 顶层输出归一化（归一化族分发容器，默认 RMSNorm）。
    pub norm: Norm,
    pub output: Linear,
    pub rope: RoPE,
    pub pad_id: u32,
}

impl MetaIModel {
    /// 构造完整模型（按 `config` 生成多层块）。
    pub fn new(config: &MetaIConfig, pad_id: u32, device: &burn::tensor::Device) -> Self {
        let embedding = EmbeddingConfig::new(config.vocab_size, config.hidden_dim).init(device);
        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(config, device));
        }
        let norm = Norm::new(config.hidden_dim, 1e-5, config.norm, device);
        let output = LinearConfig::new(config.hidden_dim, config.vocab_size)
            .with_bias(false)
            .init(device);
        // RoPE 旋转维度随注意力结构：GQA 旋整个 head_dim（Q/K 全段旋转）；
        // MLA 只旋解耦 rope 段（qk_rope_head_dim，共享单通道），latent 不旋转。
        let rope_dim = match config.attn {
            AttentionVariant::Gqa => config.hidden_dim / config.num_heads,
            AttentionVariant::Mla(cfg) => cfg.qk_rope_head_dim,
        };
        let rope = RoPE::new(rope_dim, config.max_seq_len, 10000.0, device);

        Self {
            embedding,
            blocks,
            norm,
            output,
            rope,
            pad_id,
        }
    }

    /// 模型前向：词嵌入 → 逐块算子流水线 → 顶层 Norm → 词表投影。
    pub fn forward(
        &self,
        tokens: Tensor<2, Int>,
        _mask: Option<Tensor<2, burn::tensor::Bool>>,
        cache: Option<&mut crate::infer::cache::ModelCache>,
    ) -> Tensor<3> {
        let hidden = self.hidden(tokens, _mask, cache);
        self.output.forward(hidden)
    }

    pub fn hidden(
        &self,
        tokens: Tensor<2, Int>,
        mask: Option<Tensor<2, burn::tensor::Bool>>,
        cache: Option<&mut crate::infer::cache::ModelCache>,
    ) -> Tensor<3> {
        let padding_mask = mask.unwrap_or_else(|| tokens.clone().equal_elem(self.pad_id as i32));
        self.hidden_embeddings(self.embedding.forward(tokens), padding_mask, cache)
    }

    /// Run the same causal decoder on continuous modality prefixes and token embeddings.
    pub fn hidden_embeddings(
        &self,
        mut h: Tensor<3>,
        padding_mask: Tensor<2, burn::tensor::Bool>,
        cache: Option<&mut crate::infer::cache::ModelCache>,
    ) -> Tensor<3> {
        if let Some(cache) = cache {
            for (block, layer_cache) in self.blocks.iter().zip(cache.layers.iter_mut()) {
                h = block.forward(h, &self.rope, Some(padding_mask.clone()), Some(layer_cache));
            }
        } else {
            for block in &self.blocks {
                h = block.forward(h, &self.rope, Some(padding_mask.clone()), None);
            }
        }

        self.norm.forward(h)
    }

    /// 对模型全部权重执行 int4 块量化（每个独立算子权重自动参与）。
    pub fn quantize_int4(self, _device: &burn::tensor::Device) -> Self {
        use burn::module::Quantizer;
        use burn::tensor::quantization::{Calibration, QuantScheme, QuantValue};

        let scheme = QuantScheme::default().with_value(QuantValue::Q4S);

        let mut quantizer = Quantizer::new(Calibration::MinMax, scheme);

        self.quantize_weights(&mut quantizer)
    }
}
