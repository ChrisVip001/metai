use crate::backend::{get_device, MyAutodiffBackend, MyBackend};
use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Int, Tensor};
use burn::train::metric::Adaptor;
use burn::train::metric::LossMetric;

use burn::train::LearnerBuilder;
use burn::train::{TrainOutput, TrainStep, ValidStep};

use crate::data::dpo::{DPOBatch, DPOBatcher, PreferenceDataset};
use crate::data::MetaITokenizer;
use crate::model::MetaIModel;
use crate::train::MetaITrainingConfig;

// --- DPO 损失与输出 ---

#[derive(Clone, Debug)]
pub struct DPOOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub chosen_reward: Tensor<B, 1>,
    pub rejected_reward: Tensor<B, 1>,
    pub accuracy: f32, // Policy 选出的奖励 > 被拒绝的奖励的概率
}

impl<B: Backend> Adaptor<burn::train::metric::LossInput<B>> for DPOOutput<B> {
    fn adapt(&self) -> burn::train::metric::LossInput<B> {
        burn::train::metric::LossInput::new(self.loss.clone())
    }
}

#[derive(Clone, Debug)]
pub struct DPOOutputSync {
    pub loss: f32,
    pub chosen_reward: f32,
    pub rejected_reward: f32,
    pub accuracy: f32,
}

impl<B: Backend> burn::train::metric::ItemLazy for DPOOutput<B> {
    type ItemSync = DPOOutputSync;

    fn sync(self) -> Self::ItemSync {
        DPOOutputSync {
            loss: self.loss.into_scalar().elem::<f32>(),
            chosen_reward: self.chosen_reward.into_scalar().elem::<f32>(),
            rejected_reward: self.rejected_reward.into_scalar().elem::<f32>(),
            accuracy: self.accuracy,
        }
    }
}

impl burn::train::metric::ItemLazy for DPOOutputSync {
    type ItemSync = Self;
    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: Backend> Adaptor<burn::train::metric::LossInput<B>> for DPOOutputSync {
    fn adapt(&self) -> burn::train::metric::LossInput<B> {
        burn::train::metric::LossInput::new(Tensor::from_floats([self.loss], &Default::default()))
    }
}

pub struct DPOLoss<B: Backend> {
    pub beta: f64,
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> DPOLoss<B> {
    pub fn new(beta: f64) -> Self {
        Self {
            beta,
            _b: std::marker::PhantomData,
        }
    }

    /// 计算 DPO 损失
    /// log_probs: [Batch, SeqLen] - 目标 token 的对数概率
    /// mask: [Batch, SeqLen] - 响应部分为 1，其他部分（指令/填充）为 0
    pub fn forward(
        &self,
        policy_chosen_logps: Tensor<B, 1>,
        policy_rejected_logps: Tensor<B, 1>,
        ref_chosen_logps: Tensor<B, 1>,
        ref_rejected_logps: Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, f32) {
        // 隐式奖励 = beta * (log_pi - log_ref)
        let rewards_chosen = (policy_chosen_logps.clone() - ref_chosen_logps) * self.beta;
        let rewards_rejected = (policy_rejected_logps.clone() - ref_rejected_logps) * self.beta;

        // DPO 损失 = -log sigmoid(reward_chosen - reward_rejected)
        let logits = rewards_chosen.clone() - rewards_rejected.clone();

        let loss = (-logits.clone()).exp().add_scalar(1.0).log();

        // Accuracy: chosen > rejected
        use burn::tensor::ElementConversion;
        let acc = logits
            .clone()
            .greater_elem(0.0)
            .float()
            .mean()
            .into_scalar()
            .elem();

        (
            loss.mean(),
            rewards_chosen.mean(),
            rewards_rejected.mean(),
            acc,
        )
    }

    /// 辅助函数：收集特定 token 的子集对数概率
    pub fn compute_batch_log_probs(
        logits: Tensor<B, 3>,      // [Batch, Seq, Vocab]
        labels: Tensor<B, 2, Int>, // [Batch, Seq]
        mask: Tensor<B, 2, Int>,   // [Batch, Seq]
        _pad_id: i32,
    ) -> Tensor<B, 1> {
        let [batch_size, seq_len, _vocab] = logits.dims();

        let logits_flat = logits.reshape([batch_size * seq_len, _vocab]);
        let targets_flat = labels.reshape([batch_size * seq_len]);
        let mask_flat = mask.reshape([batch_size * seq_len]); // 0 或 1

        // LogSoftmax 层
        let log_probs_flat = burn::tensor::activation::log_softmax(logits_flat, 1);

        // 收集目标 token 的对数概率
        // gather 需要 indices 形状为 [N, 1]
        let indexes = targets_flat.reshape([batch_size * seq_len, 1]);

        // Gather 结果: [N, 1]
        // 我们需要 [N]。squeeze(1) 曾报错。
        // 使用 reshape 确保形状正确。
        let token_log_probs = log_probs_flat
            .gather(1, indexes)
            .reshape([batch_size * seq_len]);

        let token_log_probs = token_log_probs * mask_flat.float();

        // 每条序列求和
        let token_log_probs = token_log_probs.reshape([batch_size, seq_len]);

        // sum_dim(1) -> [Batch, 1]
        // 展平为 [Batch]
        token_log_probs.sum_dim(1).reshape([batch_size])
    }
}

// 包装模型以同时持有 Policy 和 Ref？
// Burn 的 TrainStep trait 必须在被优化的模型上实现。
// 我们可以为 `MetaIModel` 实现 TrainStep，但它是专门针对 `DPOBatch` 的。
// 此外我们需要在 `step` 内部访问 Reference 模型。
// 选项 A: `MetaIModel` 包含一个可选的 `reference_model: Option<Arc<MetaIModel>>`。
// 选项 B: 创建一个实现 TrainStep 的 `DPOWrapper<B>` 结构体。

// 我们选择选项 B: DPOWrapper。
// 传递给 Learner 的“模型”将是 DPOWrapper。
// 但我们只想优化 `policy`。
// Burn Learner 会优化整个结构体的参数。我们需要 `reference` 不包含参数或者是被冻结的。
// 在 Burn 中，如果我们将其包装起来且不在 ADBackend 中注册，或者不输出梯度，它可能会起作用。
// 更清晰的方法：`DPOWrapper` 持有 `policy` (Module) 和 `reference` (被忽略/常量)。

#[derive(Module, Debug)]
pub struct DPOWrapper<B: Backend> {
    pub policy: MetaIModel<B>,
    // 移除了 Ignored 包装器以避免 Sync 问题。
    // 我们将手动确保梯度被分离 (detach)。
    pub reference: MetaIModel<B>,
}

impl<B: Backend> DPOWrapper<B> {
    pub fn new(policy: MetaIModel<B>, reference: MetaIModel<B>) -> Self {
        Self { policy, reference }
    }

    pub fn forward_policy(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.policy.forward(x, None, None)
    }

    pub fn forward_reference(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.reference.forward(x, None, None)
    }
}

// 为 DPOWrapper 实现 TrainStep
impl<B: AutodiffBackend> TrainStep<DPOBatch<B>, DPOOutput<B>> for DPOWrapper<B> {
    fn step(&self, batch: DPOBatch<B>) -> TrainOutput<DPOOutput<B>> {
        // 1. Policy 前向传播
        let policy_chosen_logits = self.forward_policy(batch.chosen_inputs.clone());
        let policy_rejected_logits = self.forward_policy(batch.rejected_inputs.clone());

        // 2. Reference 前向传播 (无梯度)
        let ref_chosen_logits = self.forward_reference(batch.chosen_inputs);
        let ref_rejected_logits = self.forward_reference(batch.rejected_inputs);

        // 分离 reference logits 以确保没有梯度流向那里（以防万一）
        let ref_chosen_logits = ref_chosen_logits.detach();
        let ref_rejected_logits = ref_rejected_logits.detach();

        // 3. 计算对数概率
        let pad_id = self.policy.pad_id as i32;

        let policy_chosen_logps = DPOLoss::compute_batch_log_probs(
            policy_chosen_logits,
            batch.chosen_targets.clone(),
            batch.chosen_mask.clone(),
            pad_id,
        );
        let policy_rejected_logps = DPOLoss::compute_batch_log_probs(
            policy_rejected_logits,
            batch.rejected_targets.clone(),
            batch.rejected_mask.clone(),
            pad_id,
        );

        let ref_chosen_logps = DPOLoss::compute_batch_log_probs(
            ref_chosen_logits,
            batch.chosen_targets,
            batch.chosen_mask,
            pad_id,
        );
        let ref_rejected_logps = DPOLoss::compute_batch_log_probs(
            ref_rejected_logits,
            batch.rejected_targets,
            batch.rejected_mask,
            pad_id,
        );

        // 4. 计算损失
        let dpo_loss = DPOLoss::new(0.1); // 默认 Beta = 0.1
        let (loss, c_reward, r_reward, acc) = dpo_loss.forward(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        );

        let grads = loss.backward();

        let output = DPOOutput {
            loss,
            chosen_reward: c_reward,
            rejected_reward: r_reward,
            accuracy: acc,
        };

        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> ValidStep<DPOBatch<B>, DPOOutput<B>> for DPOWrapper<B> {
    fn step(&self, batch: DPOBatch<B>) -> DPOOutput<B> {
        let policy_chosen_logits = self.forward_policy(batch.chosen_inputs.clone());
        let policy_rejected_logits = self.forward_policy(batch.rejected_inputs.clone());

        let ref_chosen_logits = self.forward_reference(batch.chosen_inputs);
        let ref_rejected_logits = self.forward_reference(batch.rejected_inputs);

        let pad_id = self.policy.pad_id as i32;

        let policy_chosen_logps = DPOLoss::compute_batch_log_probs(
            policy_chosen_logits,
            batch.chosen_targets.clone(),
            batch.chosen_mask.clone(),
            pad_id,
        );
        let policy_rejected_logps = DPOLoss::compute_batch_log_probs(
            policy_rejected_logits,
            batch.rejected_targets.clone(),
            batch.rejected_mask.clone(),
            pad_id,
        );

        let ref_chosen_logps = DPOLoss::compute_batch_log_probs(
            ref_chosen_logits,
            batch.chosen_targets,
            batch.chosen_mask,
            pad_id,
        );
        let ref_rejected_logps = DPOLoss::compute_batch_log_probs(
            ref_rejected_logits,
            batch.rejected_targets,
            batch.rejected_mask,
            pad_id,
        );

        let dpo_loss = DPOLoss::new(0.1);
        let (loss, c_reward, r_reward, acc) = dpo_loss.forward(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        );

        DPOOutput {
            loss,
            chosen_reward: c_reward,
            rejected_reward: r_reward,
            accuracy: acc,
        }
    }
}

// --- 运行器 ---

pub fn run_dpo_training(
    data_path: &str,
    model_dir: &str, // SFT 模型的检查点目录
    output_dir: &str,
) -> anyhow::Result<()> {
    let device = get_device();

    // 1. 配置
    let mut config = MetaITrainingConfig::new(crate::model::MetaIConfig::small());
    config.learning_rate = 1e-6; // DPO 需要非常小的学习率
    config.num_epochs = 1; // DPO 收敛较快
    config.batch_size = 2; // 成对数据占用更多显存
    config.tokenizer_path = "tokenizer.json".to_string();

    let tokenizer = MetaITokenizer::new(&config.tokenizer_path)?;
    let pad_id = tokenizer.pad_id().unwrap_or(0);

    // 2. 数据
    let dataset = PreferenceDataset::from_file(data_path, &tokenizer, config.model.max_seq_len)?;
    let batcher = DPOBatcher::<MyAutodiffBackend>::new(device.clone(), pad_id);
    let batcher_valid = DPOBatcher::<MyBackend>::new(device.clone(), pad_id);

    let dataloader_train = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(4)
        .build(dataset.clone());

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(4)
        .build(dataset);

    // 3. 加载模型
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();

    // 加载 Policy 模型 (被优化)
    let policy_model = MetaIModel::new(&config.model, pad_id, &device);
    let policy_model = crate::train::load_model_checkpoint(policy_model, model_dir, &device);

    // 加载 Reference 模型 (冻结)
    let ref_model = MetaIModel::new(&config.model, pad_id, &device);
    let ref_model = crate::train::load_model_checkpoint(ref_model, model_dir, &device);

    let model_wrapper = DPOWrapper::new(policy_model, ref_model);

    // 4. Learner 构建
    let learner = LearnerBuilder::new(output_dir)
        .metric_train_numeric(LossMetric::<MyAutodiffBackend>::new())
        .metric_valid_numeric(LossMetric::<MyAutodiffBackend>::new())
        .with_file_checkpointer(recorder)
        .grads_accumulation(config.grads_accumulation)
        .num_epochs(config.num_epochs)
        .build(model_wrapper, config.optimizer.init(), config.learning_rate);

    // 5. 拟合模型
    let _ = learner.fit(dataloader_train, dataloader_valid);

    Ok(())
}
