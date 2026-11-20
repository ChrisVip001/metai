//! # MultiLatentAttention —— 低秩潜空间注意力（MLA，DeepSeek-V2 结构族）
//!
//! GQA 的 KV 头扩展使 cache 随「头数 × 维度」线性增长；MLA 把**整层 KV 先压进
//! 一个低秩 latent**（每 token 一个共享向量 \(c_t\in\mathbb{R}^{r}\)），所有头的
//! Q 直接在 latent 空间与历史打分：
//!
//! \[
//! \begin{aligned}
//! c_t &= W_{DKV}\, h_t \qquad (\text{latent，cache 主体，与头数解耦})\\
//! q^{C}_{t,h} &= W_{QN,h}\, h_t \quad (\text{吸收式：DS-V2 推理侧把 }W_{UK}\text{ 折叠进 }W_Q\text{ 的等价形态})\\
//! k^{R}_{t} &= W_{KR}\, c_t \qquad (\text{解耦 RoPE 键，单通道共享})\\
//! q^{R}_{t} &= W_{QR}\, h_t \\
//! \mathrm{score}_{t,h,s} &= \frac{\langle[q^C_{t,h}; \mathrm{Rot}(q^R_t)],\ [c_s; \mathrm{Rot}(k^R_s)]\rangle}{\sqrt{r+\rho}}
//! \end{aligned}
//! \]
//!
//! 其中 \(\rho = qk\_rope\_head\_dim\)。Cache 每 token 只存
//! `[c_s; Rot(k^R_s)]`（宽 \(r+\rho\)，单通道）与 `c_s`（值侧，宽 \(r\)），
//! **与头数无关** —— 对比 GQA 的每头每维双份，长上下文显存显著更低。
//!
//! ## 与 DeepSeek-V2 的关系（诚实标注）
//!
//! - **相同**：① KV 低秩压缩为共享 latent；② 解耦 RoPE（rope 打分与每头
//!   nope 打分相加，rope 键/查询单通道共享）；③ cache 存 latent 而非解压 KV；
//! - **仓库简化（结构仍同族）**：DS-V2 训练用非吸收式（`W_{UK}/W_{UV}` 逐头
//!   升维后打分），推理优化时才吸收；本实现直接采用**吸收参数化**
//!   （q 权重已在 latent 空间，输出投影直接吃每头 latent 聚合），
//!   训练/推理同一代码路径；该参数化覆盖的函数空间与吸收后 DS-V2 相同，
//!   且省去每步对历史解压的 `O(S·H·d)` 成本；
//! - **单通道**：rope 键/查询全层共享一条（DS-V2 设计），本文件输出侧
//!   每头仍由 `w_o` 的 `Hq×r` 分块独立投影，保留头表达力。
//!
//! ## 数据流（同 GQA 的算子流水线风格）
//!
//! ```text
//! x [B, S, H]
//!  ├─ w_q_nope → split_heads            → q^C [B, Hq, S, r]
//!  ├─ w_q_rope → [B,1,S,ρ] + RoPE(off)  → q^R
//!  ├─ w_dkv → latent c3 [B,S,r]
//!  │    ├─ w_k_rope → [B,1,S,ρ] + RoPE(off) → k^R（旋转后与 c 一起入 cache）
//!  │    └─ c 本体（= v 共享通道）
//!  ├─ q = cat(q^C, broadcast q^R)       → [B, Hq, S, r+ρ]
//!  ├─ k = cat(c, k^R) 单通道 repeat Hq  → [B, Hq, S, r+ρ]   （d_k = r+ρ）
//!  ├─ v = c 单通道 repeat Hq            → [B, Hq, S, r]     （d_v = r ≠ d_k）
//!  ├─ SDPA（缩放 1/√(r+ρ)，mask 可带）
//!  ├─ merge_heads                       → [B, S, Hq·r]
//!  └─ w_o                              → [B, S, H]
//! ```
//!
//! `d_k ≠ d_v` 由 [`crate::model::sdpa`] 原生支持（softmax 只作用在打分维）。
//!
//! ## 形状与约束
//!
//! - `x`：`[B, S, Hidden]`；`rope` 以 `qk_rope_head_dim` 为旋转维构造
//!   （`RoPE::new(rope_dim, …)`，见 [`crate::model::MetaIModel::new`]）；
//! - cache 每层 K/V 为**单通道**：`k: [B, 1, L, r+ρ]`，`v: [B, 1, L, r]`
//!   （`get_full` 后对 `Hq` 做一次 repeat，与 GQA 的 KV 头扩展同机制）；
//! - MLA × KV 稀疏（DeepSeek-V3.2 路线）为后续任务：当前构造期即拒绝
//!   `config.sparse` 与 MLA 的组合（见 [`MultiLatentAttention::new`]）。
//!
//! ## 已知边界
//!
//! - 输出侧每头取 latent 聚合（`Hq·r` 进 `w_o`），非 DS 的逐头 `W_{UV}` 再混合；
//! - dense 解码走 `get_full`（无页路径；MLA×稀疏落地的页路径一并后置）；
//! - 数值等价性由「prefill（整段 causal）≡ decode（逐步 cache）」测试锁定，
//!   与 GQA 的一致性测试同一范式。

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Bool, Tensor};

use crate::model::config::MlaConfig;
use crate::model::heads::{merge_heads, repeat_kv, split_heads};
use crate::model::positional_encoding::RoPE;
use crate::model::sdpa;

/// 多潜空间注意力组合算子（MLA）。
///
/// 权重布局（吸收式，全部无 bias）：
/// - `w_q_nope`：`hidden → n_heads·kv_lora_rank`（每头直接在 latent 打分）；
/// - `w_q_rope`：`hidden → qk_rope_head_dim`（解耦 RoPE 查询，单通道）；
/// - `w_dkv`：`hidden → kv_lora_rank`（低秩压缩，cache 主体）；
/// - `w_k_rope`：`kv_lora_rank → qk_rope_head_dim`（从 latent 生成 rope 键）；
/// - `w_o`：`n_heads·kv_lora_rank → hidden`（每头 latent 聚合 → 输出）。
#[derive(Module, Debug)]
pub struct MultiLatentAttention {
    pub w_q_nope: Linear,
    pub w_q_rope: Linear,
    pub w_dkv: Linear,
    pub w_k_rope: Linear,
    pub w_o: Linear,
    pub n_heads: usize,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
}

impl MultiLatentAttention {
    /// 构造 MLA：`hidden_dim → n_heads·kv_lora_rank`（q，latent 空间）与
    /// `kv_lora_rank`（kv 压缩）与 `qk_rope_head_dim`（共享 rope）。
    pub fn new(
        hidden_dim: usize,
        n_heads: usize,
        cfg: MlaConfig,
        device: &burn::tensor::Device,
    ) -> Self {
        let rank = cfg.kv_lora_rank;
        let rope_dim = cfg.qk_rope_head_dim;
        Self {
            w_q_nope: LinearConfig::new(hidden_dim, n_heads * rank)
                .with_bias(false)
                .init(device),
            w_q_rope: LinearConfig::new(hidden_dim, rope_dim)
                .with_bias(false)
                .init(device),
            w_dkv: LinearConfig::new(hidden_dim, rank)
                .with_bias(false)
                .init(device),
            w_k_rope: LinearConfig::new(rank, rope_dim)
                .with_bias(false)
                .init(device),
            w_o: LinearConfig::new(n_heads * rank, hidden_dim)
                .with_bias(false)
                .init(device),
            n_heads,
            kv_lora_rank: rank,
            qk_rope_head_dim: rope_dim,
        }
    }

    /// MLA 前向（dense；稀疏组合在装配期被拒绝，见构造方校验）。
    ///
    /// - `x`：`[B, S, Hidden]`；
    /// - `rope`：以 `qk_rope_head_dim` 为旋转维构造的 RoPE；
    /// - `mask`：padding mask（仅无 cache 时生效，因果在内部叠加）；
    /// - `cache`：可选 KV Cache（存 `[c; Rot(k^R)]` 与 `c`，单通道）。
    pub fn forward(
        &self,
        x: Tensor<3>,
        rope: &RoPE,
        mask: Option<Tensor<2, Bool>>,
        cache: Option<&mut crate::infer::cache::KVCache>,
    ) -> Tensor<3> {
        let [batch_size, seq_len, _hidden] = x.dims();
        let rank = self.kv_lora_rank;
        let rope_dim = self.qk_rope_head_dim;
        let n_heads = self.n_heads;
        let scale = ((rank + rope_dim) as f32).sqrt().recip();

        // ============ 投影 ============
        // q^C 每头 [B,Hq,S,r]；q^R / k^R 单通道 [B,1,S,ρ]
        let q_c = split_heads(self.w_q_nope.forward(x.clone()), n_heads);
        let q_r3 = self.w_q_rope.forward(x.clone()); // [B,S,ρ]
        let c3 = self.w_dkv.forward(x); // [B,S,r]（latent）

        // ============ RoPE（偏移 = cache 长度，解码主路径） ============
        let offset = cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);
        let q_r = rope.forward_with_offset::<4>(q_r3.unsqueeze_dim::<4>(1), offset); // [B,1,S,ρ]
        let k_r = rope.forward_with_offset::<4>(
            self.w_k_rope.forward(c3.clone()).unsqueeze_dim::<4>(1),
            offset,
        ); // [B,1,S,ρ]（旋转后入 cache / 打分）

        // latent 作共享键/值通道：[B,1,S,r]
        let c4 = c3.unsqueeze_dim::<4>(1);

        // q 拼接 rope 段并广播到每头：[B,Hq,S,r+ρ]
        let q_r_heads = q_r.clone().expand([batch_size, n_heads, seq_len, rope_dim]);
        let q = Tensor::cat(vec![q_c, q_r_heads], 3);

        if let Some(cache) = cache {
            let padding = cache.append_padding(mask, batch_size, seq_len, &q.device());
            // ============ Cache 更新（单通道：k = [c; k^R]，v = c） ============
            let k4 = Tensor::cat(vec![c4.clone(), k_r], 3); // [B,1,1,r+ρ]
            cache.update(k4, c4);

            // ============ SDPA（dense 解码：天然因果，无 mask） ============
            let (k_full, v_full) = cache.get_full();
            let k = repeat_kv(k_full, n_heads);
            let v = repeat_kv(v_full, n_heads);
            let out = {
                let mask = sdpa::cached_mask(offset, seq_len, &q.device(), padding);
                sdpa::scaled_dot_product_attention(q, k, v, scale, mask)
            };

            // ============ 合并头 + 输出投影 ============
            let out = merge_heads(out); // [B,S,Hq·r]
            self.w_o.forward(out)
        } else {
            // k = [c; k^R]（单通道 [B,1,S,r+ρ]）；v = c
            let k = Tensor::cat(vec![c4.clone(), k_r], 3);
            let v = c4;
            let k = repeat_kv(k, n_heads);
            let v = repeat_kv(v, n_heads);

            // mask 组装（因果 + 可选 padding；dense 路径）
            let attn_mask: Option<Tensor<4, Bool>> = if seq_len > 1 {
                let device = q.device();
                let causal =
                    sdpa::build_causal_mask(seq_len, &device).reshape([1, 1, seq_len, seq_len]);
                match mask {
                    Some(pad_mask) => {
                        let pad_expanded = pad_mask
                            .unsqueeze::<4>()
                            .reshape([batch_size, 1, 1, seq_len]);
                        Some(sdpa::combine_masks(
                            causal,
                            pad_expanded,
                            batch_size,
                            seq_len,
                        ))
                    }
                    None => Some(causal),
                }
            } else {
                None
            };

            let out = sdpa::scaled_dot_product_attention(q, k, v, scale, attn_mask);
            let out = merge_heads(out); // [B,S,Hq·r]
            self.w_o.forward(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::get_device;
    use crate::infer::cache::KVCache;
    use crate::model::config::MlaConfig;

    fn mk_attn(hidden: usize, heads: usize) -> MultiLatentAttention {
        let device = get_device();
        MultiLatentAttention::new(hidden, heads, MlaConfig::new(8, 4), &device)
    }

    /// prefill（整段 + causal）≡ 逐 token decode（cache）：锁 RoPE offset 与
    /// latent 缓存的逐位置语义（MLA 核心等价性）。
    #[test]
    fn test_mla_prefill_matches_decode() {
        let device = get_device();
        let hidden = 16;
        let heads = 4;
        let rope_dim = 4;

        let attn = mk_attn(hidden, heads);
        let rope = RoPE::new(rope_dim, 128, 10000.0, &device);

        let seq_len = 7;
        let input = Tensor::<3>::random(
            [1, seq_len, hidden],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out_prefill = attn.forward(input.clone(), &rope, None, None);

        let mut cache = KVCache::new();
        let mut parts = Vec::new();
        for i in 0..seq_len {
            let step = input.clone().slice([0..1, i..i + 1, 0..hidden]);
            parts.push(attn.forward(step, &rope, None, Some(&mut cache)));
        }
        let out_decode = Tensor::cat(parts, 1);

        let diff = (out_prefill - out_decode).abs().max().into_scalar::<f32>();
        println!("MLA prefill vs decode Max Diff: {diff}");
        assert!(
            diff < 1e-4,
            "MLA prefill 应与逐步 decode 一致，diff = {diff}"
        );
    }

    /// cache 形态断言：k 单通道宽 `r+ρ`、v 单通道宽 `r`（压缩语义生效）。
    #[test]
    fn test_mla_cache_stores_latent_channel() {
        let device = get_device();
        let hidden = 16;
        let heads = 4;
        let (rank, rope_dim) = (8usize, 4usize);

        let attn =
            MultiLatentAttention::new(hidden, heads, MlaConfig::new(rank, rope_dim), &device);
        let rope = RoPE::new(rope_dim, 128, 10000.0, &device);

        let input = Tensor::<3>::random(
            [1, 3, hidden],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mut cache = KVCache::new();
        for i in 0..3 {
            let step = input.clone().slice([0..1, i..i + 1, 0..hidden]);
            let _ = attn.forward(step, &rope, None, Some(&mut cache));
        }

        let (k_full, v_full) = cache.get_full();
        assert_eq!(
            k_full.dims(),
            [1, 1, 3, rank + rope_dim],
            "k 应为单通道 [c;k^R]"
        );
        assert_eq!(v_full.dims(), [1, 1, 3, rank], "v 应为单通道 latent");
    }

    /// padding mask 路径可用（无 cache、多行 + mask 不 panic、形状正确）。
    #[test]
    fn test_mla_with_padding_mask_forward() {
        let device = get_device();
        let hidden = 16;
        let heads = 2;

        let attn = mk_attn(hidden, heads);
        let rope = RoPE::new(4, 128, 10000.0, &device);

        let input = Tensor::<3>::random(
            [2, 5, hidden],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        // pad 位：第二行最后 2 个 token（真实 mask 由上层用 pad_id 生成）
        let pad = Tensor::<2, Bool>::from_data(
            burn::tensor::TensorData::new(
                vec![
                    false, false, false, false, false, false, false, false, true, true,
                ],
                [2, 5],
            ),
            &device,
        );
        let out = attn.forward(input, &rope, Some(pad), None);
        assert_eq!(out.dims(), [2, 5, hidden]);
    }
}
