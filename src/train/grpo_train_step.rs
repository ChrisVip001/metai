use crate::data::sft::SFTBatch;
use crate::model::MetaIModel;
use crate::train::grpo::{GRPOConfig, GRPOLoss};
use crate::train::MetaIOutput;
use burn::module::Module;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::{TrainOutput, TrainStep, ValidStep};

/// GRPO 训练封装器，持有 Policy 和 Reference 模型
#[derive(Module, Debug)]
pub struct GRPOTrainWrapper<B: Backend> {
    pub policy: MetaIModel<B>,
    pub reference: MetaIModel<B>,
    pub group_size: usize,
}

impl<B: Backend> GRPOTrainWrapper<B> {
    pub fn new(policy: MetaIModel<B>, reference: MetaIModel<B>, group_size: usize) -> Self {
        Self {
            policy,
            reference,
            group_size,
        }
    }
}

// 适配 Burn 的 TrainStep 接口
impl<B: AutodiffBackend> TrainStep<SFTBatch<B>, MetaIOutput<B>> for GRPOTrainWrapper<B> {
    fn step(&self, batch: SFTBatch<B>) -> TrainOutput<MetaIOutput<B>> {
        let [total_samples, seq_len] = batch.inputs.dims();
        let batch_size = total_samples / self.group_size;

        // 1. Policy Forward
        let policy_logits = self.policy.forward(batch.inputs.clone(), None, None);

        // 2. Reference Forward (No Grad)
        let ref_logits = self.reference.forward(batch.inputs.clone(), None, None);
        let ref_logits = ref_logits.detach();

        // 3. 计算 log probs (针对 target tokens)
        let policy_logps_all = burn::tensor::activation::log_softmax(policy_logits, 2);
        let ref_logps_all = burn::tensor::activation::log_softmax(ref_logits, 2);

        // 获取每个 token 对应的 log prob
        let indices = batch.targets.clone().reshape([total_samples, seq_len, 1]);

        let policy_logps = policy_logps_all.gather(2, indices.clone()).squeeze::<2>();
        let ref_logps = ref_logps_all.gather(2, indices).squeeze::<2>();

        // 重新排列为 [Batch, Group, Seq]
        let policy_logps = policy_logps.reshape([batch_size, self.group_size, seq_len]);
        let ref_logps = ref_logps.reshape([batch_size, self.group_size, seq_len]);

        let mask = batch.mask.reshape([batch_size, self.group_size, seq_len]);

        // 构造虚拟 Reward (在正式集成中应由 RewardModel 产生)
        let rewards = burn::tensor::Tensor::<B, 2>::zeros(
            [batch_size, self.group_size],
            &batch.inputs.device(),
        );

        let grpo_loss_fn = GRPOLoss::<B>::new(GRPOConfig::new());
        let loss = grpo_loss_fn.forward(policy_logps, ref_logps, rewards, mask);

        let grads = loss.backward();
        TrainOutput::new(self, grads, MetaIOutput { loss })
    }
}

impl<B: Backend> ValidStep<SFTBatch<B>, MetaIOutput<B>> for GRPOTrainWrapper<B> {
    fn step(&self, batch: SFTBatch<B>) -> MetaIOutput<B> {
        let [total_samples, seq_len] = batch.inputs.dims();
        let batch_size = total_samples / self.group_size;

        let policy_logits = self.policy.forward(batch.inputs.clone(), None, None);
        let ref_logits = self.reference.forward(batch.inputs.clone(), None, None);

        let policy_logps_all = burn::tensor::activation::log_softmax(policy_logits, 2);
        let ref_logps_all = burn::tensor::activation::log_softmax(ref_logits, 2);

        let indices = batch.targets.reshape([total_samples, seq_len, 1]);

        let policy_logps = policy_logps_all.gather(2, indices.clone()).squeeze::<2>();
        let ref_logps = ref_logps_all.gather(2, indices).squeeze::<2>();

        let policy_logps = policy_logps.reshape([batch_size, self.group_size, seq_len]);
        let ref_logps = ref_logps.reshape([batch_size, self.group_size, seq_len]);

        let mask = batch.mask.reshape([batch_size, self.group_size, seq_len]);
        let rewards = burn::tensor::Tensor::<B, 2>::zeros(
            [batch_size, self.group_size],
            &policy_logps.device(),
        );

        let grpo_loss_fn = GRPOLoss::<B>::new(GRPOConfig::new());
        let loss = grpo_loss_fn.forward(policy_logps, ref_logps, rewards, mask);

        MetaIOutput { loss }
    }
}
