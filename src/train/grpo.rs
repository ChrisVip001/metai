//! GRPO 损失与训练超参数配置
//!
//! - [`GRPOLoss`]：组内归一化优势 + KL(pi||ref) 惩罚（单步 Vanilla Policy Gradient）。
//! - [`GRPOConfig`]：rollout / 采样 / 奖励 的完整超参数（CLI 与训练共用）。
//!
//! 真正的训练入口（rollout -> 规则奖励 -> 手动优化循环）位于 `grpo_train_step.rs`。

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::train::reward::RewardKind;

/// GRPO 训练超参数。
///
/// 注意：此处不用 `burn::config::Config` 派生，因为字段包含 `RewardKind` 这类
/// 自定义枚举，`#[config(default = ...)]` 对其支持不稳定；改用手写 `Default`。
#[derive(Clone, Debug, PartialEq)]
pub struct GRPOConfig {
    /// KL 惩罚系数。
    pub beta: f64,
    /// 裁剪系数（保留字段，供将来 PPO-clip 变体使用）。
    pub clip_eps: f64,
    /// 每个 prompt 采样的响应数（组内基线要求 >= 2）。
    pub group_size: usize,
    /// rollout 最大新 token 数。
    pub max_new_tokens: usize,
    /// 规则奖励类型。
    pub reward: RewardKind,
    /// 采样温度。
    pub temperature: f32,
    /// Top-K 采样候选数。
    pub top_k: usize,
    /// Top-P 累积概率截断。
    pub top_p: f32,
}

impl Default for GRPOConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            clip_eps: 0.2,
            group_size: 4,
            max_new_tokens: 96,
            reward: RewardKind::Rule,
            temperature: 0.9,
            top_k: 50,
            top_p: 0.95,
        }
    }
}

pub struct GRPOLoss<B: Backend> {
    pub config: GRPOConfig,
    pub _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> GRPOLoss<B> {
    pub fn new(config: GRPOConfig) -> Self {
        Self {
            config,
            _marker: std::marker::PhantomData,
        }
    }

    /// 计算 GRPO Loss
    ///
    /// # Arguments
    /// * `policy_logprobs`: [Batch, GroupSize, SeqLen] - Policy Model 的 Log Probabilities
    /// * `ref_logprobs`: [Batch, GroupSize, SeqLen] - Reference Model 的 Log Probabilities
    /// * `rewards`: [Batch, GroupSize] - 每个样本的奖励
    /// * `mask`: [Batch, GroupSize, SeqLen] - Padding Mask
    pub fn forward(
        &self,
        policy_logprobs: Tensor<B, 3>,
        ref_logprobs: Tensor<B, 3>,
        rewards: Tensor<B, 2>,
        mask: Tensor<B, 3, Int>,
    ) -> Tensor<B, 1> {
        let [batch_size, group_size, seq_len] = policy_logprobs.dims();
        let _device = policy_logprobs.device();

        // 1. 计算优势 (Advantage)
        // Group 内标准化: A_i = (r_i - mean(r)) / (std(r) + epsilon)
        let mean_rewards = rewards.clone().mean_dim(1).reshape([batch_size, 1]);
        let std_rewards = rewards.clone().var(1).sqrt().reshape([batch_size, 1]);
        let advantages = (rewards - mean_rewards) / (std_rewards + 1e-8);

        // 广播 Advantage 到序列维度 [Batch, Group, 1] -> [Batch, Group, SeqLen]
        let advantages = advantages
            .reshape([batch_size, group_size, 1])
            .expand([batch_size, group_size, seq_len]);

        // 2. 简化 GRPO：带组基线的 Policy Gradient + KL 惩罚
        // 目标：最大化 (log_pi * A - beta * KL(pi || ref))
        // 损失：最小化 -(log_pi * A - beta * KL)
        let kl = policy_logprobs.clone() - ref_logprobs;
        let token_loss = (policy_logprobs * advantages) - (kl * self.config.beta);

        // Mask 遮罩
        let mask = mask.float();
        let token_loss = token_loss * mask.clone();

        // 仅在有效 token 上平均
        let loss = -token_loss.sum() / (mask.sum() + 1e-8);

        loss.reshape([1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{get_device, MyBackend};

    type TestBackend = MyBackend;

    #[test]
    fn test_grpo_loss_basic() {
        let device = get_device();
        let config = GRPOConfig::default();
        let grpo = GRPOLoss::<TestBackend>::new(config);

        // Batch=1, Group=2, Seq=3
        let batch_size = 1;
        let group_size = 2;
        let seq_len = 3;

        // 1. 奖励：组内 [1.0, 2.0] -> Adv0 < 0, Adv1 > 0
        let rewards = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0]], &device);

        // 2. Policy/Ref Logprobs 全 0 => KL=0
        let policy_lp = Tensor::<TestBackend, 3>::zeros([batch_size, group_size, seq_len], &device);
        let ref_lp = Tensor::<TestBackend, 3>::zeros([batch_size, group_size, seq_len], &device);
        let mask = Tensor::<TestBackend, 3, Int>::ones([batch_size, group_size, seq_len], &device);

        // Loss = -(mean(log_pi * A - beta * 0)) = 0
        let loss = grpo.forward(policy_lp.clone(), ref_lp.clone(), rewards.clone(), mask.clone());
        let loss_val = loss.into_scalar();
        assert!(loss_val.abs() < 1e-5);
    }

    #[test]
    fn test_grpo_loss_prefers_better_response() {
        let device = get_device();
        let config = GRPOConfig::default();
        let grpo = GRPOLoss::<TestBackend>::new(config);

        let batch_size = 1;
        let group_size = 2;
        let seq_len = 3;

        let rewards = Tensor::<TestBackend, 2>::from_floats([[1.0, -1.0]], &device);
        let ref_lp = Tensor::<TestBackend, 3>::zeros([batch_size, group_size, seq_len], &device);
        let mask = Tensor::<TestBackend, 3, Int>::ones([batch_size, group_size, seq_len], &device);

        let run = |policy_lp: Tensor<TestBackend, 3>| -> f32 {
            grpo.forward(policy_lp, ref_lp.clone(), rewards.clone(), mask.clone())
                .into_scalar()
        };

        // 好：policy 给高奖励样本(row0)更高概率 -> 该方向 loss 更低
        let good = Tensor::<TestBackend, 3>::from_floats(
            [[[-0.1, -0.1, -0.1], [-1.0, -1.0, -1.0]]],
            &device,
        );
        let good_loss = run(good);

        // 坏：policy 给低奖励样本(row1)更高概率
        let bad = Tensor::<TestBackend, 3>::from_floats(
            [[[-1.0, -1.0, -1.0], [-0.1, -0.1, -0.1]]],
            &device,
        );
        let bad_loss = run(bad);

        assert!(
            good_loss < bad_loss,
            "policy 应倾向高奖励样本: good={good_loss}, bad={bad_loss}"
        );
    }
}
