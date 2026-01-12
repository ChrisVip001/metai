use crate::backend::{get_device, MyAutodiffBackend, MyBackend};
use crate::data::MetaITokenizer;
use crate::model::MetaIModel;
use crate::train::MetaITrainingConfig;
use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn::train::metric::LossMetric;
use burn::train::LearnerBuilder;

#[derive(Config, Debug)]
pub struct GRPOConfig {
    #[config(default = 0.1)]
    pub beta: f64, // KL penalty coefficient
    #[config(default = 0.2)]
    pub clip_eps: f64, // PPO clipping epsilon
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

        // 2. 计算 Ratio (Importance Sampling)
        // ratio = exp(log_pi - log_ref) = pi / ref
        // 注意：这里假设 policy_logprobs 是相对于 old_policy 的 (PPO style)，
        // 但 GRPO 通常简化为相对于 Reference 或只做单步更新。
        // 如果是单步 GRPO (如 DeepSeekMath)，通常直接优化 log_pi * A - beta * KL
        // DeepSeek-V3 论文公式: E [ (pi/pi_old) * A ] ... ?
        // 实际上 GRPO 往往结合 PPO:
        // ratio = exp(log_probs - old_log_probs)
        // 这里为了简化，我们假设 old_policy == ref_model (即第一步) 或者我们维护了 old_log_probs。
        // 在此实现中，我们计算 log_pi * A - beta * KL(pi || ref) 的简化形式 (Vanilla Policy Gradient with Group Baseline)
        // 或者实现完整的 PPO-GRPO。

        // 我们实现带 KL 惩罚的 Policy Gradient: LOSS = - (mean(log_pi * A) - beta * KL)

        // KL(pi || ref) approx = log_pi - log_ref
        let kl = policy_logprobs.clone() - ref_logprobs;

        // Per-token loss term
        // Objective: maximize (log_pi * A - beta * KL)
        // Loss: minimize -(log_pi * A - beta * KL)
        let token_loss = (policy_logprobs * advantages) - (kl * self.config.beta);

        // Masking
        let mask = mask.float();
        let token_loss = token_loss * mask.clone();

        // Average over valid tokens
        let loss = -token_loss.sum() / (mask.sum() + 1e-8);

        loss.reshape([1])
    }
}

pub fn run_grpo_training(data_path: &str, model_dir: &str, output_dir: &str) -> anyhow::Result<()> {
    let device = get_device();
    let group_size = 4;

    // 1. Config
    let mut config = MetaITrainingConfig::new(crate::model::MetaIConfig::small());
    config.learning_rate = 5e-7;
    config.num_epochs = 1;
    config.batch_size = 2; // Batch of groups
    config.tokenizer_path = "tokenizer.json".to_string();

    let tokenizer = MetaITokenizer::new(&config.tokenizer_path)?;
    let pad_id = tokenizer.pad_id().unwrap_or(0);

    // 2. Data (Reuse SFT logic but interpreted as Groups)
    let dataset = crate::data::sft::InstructionDataset::from_file(
        data_path,
        &tokenizer,
        config.model.max_seq_len,
    )?;

    // 3. Load Policy & Reference
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();

    let policy_model = MetaIModel::new(&config.model, pad_id, &device);
    let ref_model = MetaIModel::new(&config.model, pad_id, &device);

    let policy_model = if let Some(epoch) = crate::train::train::find_latest_epoch(model_dir) {
        println!("Loading Policy from epoch {}", epoch);
        let model_path = std::path::Path::new(model_dir)
            .join("checkpoint")
            .join(format!("model-{}.bin", epoch));
        policy_model
            .load_file(model_path, &recorder, &device)
            .expect("Failed to load policy")
    } else {
        policy_model
    };

    let ref_model = if let Some(epoch) = crate::train::train::find_latest_epoch(model_dir) {
        let model_path = std::path::Path::new(model_dir)
            .join("checkpoint")
            .join(format!("model-{}.bin", epoch));
        ref_model
            .load_file(model_path, &recorder, &device)
            .expect("Failed to load ref")
    } else {
        ref_model
    };

    // 4. Wrapper & Learner
    let wrapper =
        crate::train::grpo_train_step::GRPOTrainWrapper::new(policy_model, ref_model, group_size);

    let batcher = crate::data::sft::SFTBatcher::<MyAutodiffBackend>::new(device.clone(), pad_id);
    let batcher_valid = crate::data::sft::SFTBatcher::<MyBackend>::new(device.clone(), pad_id);

    let dataloader_train = burn::data::dataloader::DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size * group_size)
        .shuffle(config.seed)
        .num_workers(4)
        .build(dataset.clone());

    let dataloader_valid = burn::data::dataloader::DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size * group_size)
        .num_workers(4)
        .build(dataset);

    let learner = LearnerBuilder::new(output_dir)
        .metric_train_numeric(LossMetric::<MyAutodiffBackend>::new())
        .metric_valid_numeric(LossMetric::<MyAutodiffBackend>::new())
        .with_file_checkpointer(recorder)
        .num_epochs(config.num_epochs)
        .build(wrapper, config.optimizer.init(), config.learning_rate);

    let _ = learner.fit(dataloader_train, dataloader_valid);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{get_device, MyBackend};

    type TestBackend = MyBackend;

    #[test]
    fn test_grpo_loss_basic() {
        let device = get_device();
        let config = GRPOConfig::new();
        let grpo = GRPOLoss::<TestBackend>::new(config);

        // Batch=1, Group=2, Seq=3
        let batch_size = 1;
        let group_size = 2;
        let seq_len = 3;

        // 1. Rewards: Group 0 -> 1.0, Group 1 -> 2.0
        // Mean = 1.5, Std = 0.5 (approx, population vs sample var differs but logic holds)
        // Adv 0 < 0, Adv 1 > 0
        let rewards = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0]], &device);

        // 2. Policy Logprobs (Close to Ref)
        let policy_lp = Tensor::<TestBackend, 3>::zeros([batch_size, group_size, seq_len], &device);
        let ref_lp = Tensor::<TestBackend, 3>::zeros([batch_size, group_size, seq_len], &device);

        let mask = Tensor::<TestBackend, 3, Int>::ones([batch_size, group_size, seq_len], &device);

        // Loss = - (Mean(log_pi * A - beta * 0)) = - Mean(0 * A) = 0
        let loss = grpo.forward(
            policy_lp.clone(),
            ref_lp.clone(),
            rewards.clone(),
            mask.clone(),
        );

        // Assert Loss is near 0
        let loss_val = loss.into_scalar();
        assert!(loss_val.abs() < 1e-5);
    }
}
