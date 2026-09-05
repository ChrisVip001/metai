//! GRPO rollout：策略自回归采样（真正的 rollout，替代旧的"拿 SFT targets 冒充 rollout"）
//!
//! 采样逻辑在 CPU 侧（top-k / top-p / 温度），便于单元测试；
//! 每步只调用一次模型前向取得下一个 token 的 logits。

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use rand::Rng;

use crate::model::MetaIModel;

#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 0.95,
        }
    }
}

/// 从一组 logits 中做 top-k + top-p + 温度采样（纯 CPU，可单测）
pub fn sample_token_from_logits(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut impl Rng,
) -> u32 {
    debug_assert!(!logits.is_empty());
    let temp = if temperature > 0.0 { temperature } else { 1.0 };

    // 温度缩放
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();
    let max_l = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // 候选池
    let mut idx: Vec<usize> = (0..scaled.len()).collect();
    idx.sort_by(|&a, &b| {
        scaled[b]
            .partial_cmp(&scaled[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let k = if top_k > 0 {
        top_k.min(idx.len())
    } else {
        idx.len()
    };
    idx.truncate(k);

    // top-p：从高到低累计概率
    if top_p > 0.0 && top_p < 1.0 {
        let mut cum = 0.0f64;
        let mut cut = idx.len();
        for (j, &i) in idx.iter().enumerate() {
            cum += ((scaled[i] - max_l) as f64).exp();
            if cum >= top_p as f64 {
                cut = j + 1;
                break;
            }
        }
        idx.truncate(cut);
    }

    debug_assert!(!idx.is_empty());
    // 软最大权重采样
    let weights: Vec<f64> = idx.iter().map(|&i| ((scaled[i] - max_l) as f64).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut r: f64 = rng.gen::<f64>() * total;
    let mut chosen = idx[0];
    for (j, &w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            chosen = idx[j];
            break;
        }
    }
    chosen as u32
}

/// 单条 rollout：以 prompt 为条件，自回归采样至多 max_new_tokens 个 token，
/// 遇 eos（不含 eos）或达上限停止。
///
/// 注意：模型前向在 autodiff 后端上会构建计算图，本函数会在每步结果 drop 后
/// 由后端自动回收；训练中无需对 rollout 结果求梯度（梯度在重算 logprob 时获得）。
pub fn rollout_one<B: Backend>(
    model: &MetaIModel<B>,
    device: &B::Device,
    prompt: &[u32],
    max_new_tokens: usize,
    sampling: &SamplingConfig,
    eos_id: u32,
    rng: &mut impl Rng,
) -> Vec<u32> {
    if prompt.is_empty() || max_new_tokens == 0 {
        return Vec::new();
    }

    let mut ctx: Vec<u32> = prompt.to_vec();
    let mut out: Vec<u32> = Vec::with_capacity(max_new_tokens);

    while out.len() < max_new_tokens {
        let ids: Vec<i32> = ctx.iter().map(|&t| t as i32).collect();
        let input = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(ids, [1, ctx.len()]),
            device,
        );

        // 取出最后一列的 logits 到 CPU
        let logits = model.forward(input, None, None);
        let [_, seq, vocab] = logits.dims();
        let row: Vec<f32> = logits
            .slice([0..1, (seq - 1)..seq, 0..vocab])
            .flatten::<1>(0, 2)
            .into_data()
            .iter::<f32>()
            .collect();

        let next = sample_token_from_logits(
            &row,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            rng,
        );

        if next == eos_id {
            break; // 停止符不入序列（与数据构造保持一致）
        }
        ctx.push(next);
        out.push(next);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_sample_deterministic_top1() {
        // top_k=1 时必然返回最大 logits 的 token
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let logits = vec![1.0, 5.0, -2.0, 3.0];
        let t = sample_token_from_logits(&logits, 1.0, 1, 0.0, &mut rng);
        assert_eq!(t, 1); // index 1 最大
    }

    #[test]
    fn test_sample_top_p_never_empty() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let logits: Vec<f32> = (0..64).map(|i| i as f32 / 10.0).collect();
        for _ in 0..50 {
            let t = sample_token_from_logits(&logits, 0.8, 20, 0.9, &mut rng);
            assert!(t < 64);
        }
    }

    #[test]
    fn test_sample_low_temp_prefers_max() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let logits = vec![0.0, 10.0, 0.0, 0.0, 0.0];
        let mut counts = [0usize; 5];
        for _ in 0..200 {
            let t = sample_token_from_logits(&logits, 0.05, 0, 0.0, &mut rng);
            counts[t as usize] += 1;
        }
        assert!(counts[1] > 190, "低温应基本落在 argmax，got {counts:?}");
    }
}
