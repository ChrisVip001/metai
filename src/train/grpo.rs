//! Token- and sequence-level policy objectives. Sampling probabilities are frozen for both update passes.
use burn::tensor::Tensor;
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq)]
pub enum Algorithm {
    Ppo,
    Grpo,
    Gspo,
    Cispo,
}

pub fn advantages(rewards: &[f32]) -> Vec<f32> {
    let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
    let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rewards.len() as f32;
    rewards
        .iter()
        .map(|r| (r - mean) / (variance.sqrt() + 1e-8))
        .collect()
}

#[allow(clippy::too_many_arguments)] // Each tensor corresponds to a term in the policy objective.
pub fn policy_loss(
    method: Algorithm,
    current: Tensor<2>,
    old: Tensor<2>,
    reference: Tensor<2>,
    advantage: Tensor<2>,
    mask: Tensor<2>,
    clip: f64,
    beta: f64,
) -> Tensor<1> {
    if method == Algorithm::Gspo {
        // Equal weight per response, independent of the number of supervised positions.
        let lengths = mask.clone().sum_dim(1).clamp_min(1.);
        let response_mean = |x: Tensor<2>| (x * mask.clone()).sum_dim(1) / lengths.clone();
        let ratio = response_mean(current.clone() - old.detach()).exp();
        let a = response_mean(advantage);
        let bounded = ratio.clone().clamp(1. - clip, 1. + clip);
        let gain = (ratio * a.clone()).min_pair(bounded * a);
        let delta = reference.detach() - current;
        return (response_mean(delta.clone().exp() - delta - 1.) * beta - gain).mean();
    }
    let ratio = (current.clone() - old.detach()).exp();
    let bounded = ratio.clone().clamp(1. - clip, 1. + clip);
    let gain = match method {
        Algorithm::Cispo => bounded.detach() * advantage * current.clone(),
        _ => (ratio * advantage.clone()).min_pair(bounded * advantage),
    };
    let delta = reference.detach() - current;
    let kl = delta.clone().exp() - delta - 1.;
    ((kl * beta - gain) * mask.clone()).sum() / mask.sum()
}
