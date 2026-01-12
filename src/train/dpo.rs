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

// --- DPO Loss & Output ---

#[derive(Clone, Debug)]
pub struct DPOOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub chosen_reward: Tensor<B, 1>,
    pub rejected_reward: Tensor<B, 1>,
    pub accuracy: f32, // Policy chosen > rejected
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

    /// Compute DPO Loss
    /// log_probs: [Batch, SeqLen] - Log probabilities of the TARGET tokens
    /// mask: [Batch, SeqLen] - 1 for valid tokens (Response), 0 for others (Instruction/Pad)
    pub fn forward(
        &self,
        policy_chosen_logps: Tensor<B, 1>,
        policy_rejected_logps: Tensor<B, 1>,
        ref_chosen_logps: Tensor<B, 1>,
        ref_rejected_logps: Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, f32) {
        // implicit reward = beta * (log_pi - log_ref)
        let rewards_chosen = (policy_chosen_logps.clone() - ref_chosen_logps) * self.beta;
        let rewards_rejected = (policy_rejected_logps.clone() - ref_rejected_logps) * self.beta;

        // DPO Loss = -log sigmoid(reward_chosen - reward_rejected)
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

    /// Helper to gather log probabilities of specific tokens
    pub fn compute_batch_log_probs(
        logits: Tensor<B, 3>,      // [Batch, Seq, Vocab]
        labels: Tensor<B, 2, Int>, // [Batch, Seq]
        mask: Tensor<B, 2, Int>,   // [Batch, Seq]
        _pad_id: i32,
    ) -> Tensor<B, 1> {
        let [batch_size, seq_len, _vocab] = logits.dims();

        let logits_flat = logits.reshape([batch_size * seq_len, _vocab]);
        let targets_flat = labels.reshape([batch_size * seq_len]);
        let mask_flat = mask.reshape([batch_size * seq_len]); // 0 or 1

        // LogSoftmax
        let log_probs_flat = burn::tensor::activation::log_softmax(logits_flat, 1);

        // Gather target log probs
        // gather needs indices [N, 1]
        let indexes = targets_flat.reshape([batch_size * seq_len, 1]);

        // gather result: [N, 1]
        // We want [N]. squeeze(1) was failing.
        // Use reshape ensures shape is correct.
        let token_log_probs = log_probs_flat
            .gather(1, indexes)
            .reshape([batch_size * seq_len]);

        let token_log_probs = token_log_probs * mask_flat.float();

        // Sum per sequence
        let token_log_probs = token_log_probs.reshape([batch_size, seq_len]);

        // sum_dim(1) -> [Batch, 1]
        // Flatten to [Batch]
        token_log_probs.sum_dim(1).reshape([batch_size])
    }
}

// Wrapper Model to hold Policy and Ref?
// Burn's TrainStep trait must be implemented on the Model being optimized.
// We can implement TrainStep for `MetaIModel` but specifically for `DPOBatch`.
// But we need the Reference model inside `step`.
// Option A: `MetaIModel` has an optional `reference_model: Option<Arc<MetaIModel>>`.
// Option B: Create a `DPOWrapper<B>` struct that implements TrainStep.

// Let's go with Option B: DPOWrapper.
// The "Model" passed to Learner will be DPOWrapper.
// But we want to optimize `policy`.
// Burn Learner optimizes the whole struct params. We need `reference` to be non-param or frozen.
// In Burn, if we wrap it and don't register it in `ADBackend` or just don't output gradients, it might work.
// Cleaner way: `DPOWrapper` holds `policy` (Module) and `reference` (Ignored/Constant).

#[derive(Module, Debug)]
pub struct DPOWrapper<B: Backend> {
    pub policy: MetaIModel<B>,
    // Removed Ignored wrapper to avoid Sync issues.
    // We will ensure gradients are detached manually.
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

// Implement TrainStep for DPOWrapper
impl<B: AutodiffBackend> TrainStep<DPOBatch<B>, DPOOutput<B>> for DPOWrapper<B> {
    fn step(&self, batch: DPOBatch<B>) -> TrainOutput<DPOOutput<B>> {
        // 1. Policy Forward
        let policy_chosen_logits = self.forward_policy(batch.chosen_inputs.clone());
        let policy_rejected_logits = self.forward_policy(batch.rejected_inputs.clone());

        // 2. Reference Forward (No Grad)
        let ref_chosen_logits = self.forward_reference(batch.chosen_inputs);
        let ref_rejected_logits = self.forward_reference(batch.rejected_inputs);

        // Detach reference logits to ensure no grad flows there (just in case)
        let ref_chosen_logits = ref_chosen_logits.detach();
        let ref_rejected_logits = ref_rejected_logits.detach();

        // 3. Compute Log Probs
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

        // 4. Loss
        let dpo_loss = DPOLoss::new(0.1); // Beta = 0.1 default
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

// --- Runner ---

pub fn run_dpo_training(
    data_path: &str,
    model_dir: &str, // Checkpoint dir of SFT model
    output_dir: &str,
) -> anyhow::Result<()> {
    let device = get_device();

    // 1. Config
    let mut config = MetaITrainingConfig::new(crate::model::MetaIConfig::small());
    config.learning_rate = 1e-6; // DPO needs very small LR
    config.num_epochs = 1; // DPO converges fast
    config.batch_size = 2; // Pairs take more memory
    config.tokenizer_path = "tokenizer.json".to_string();

    let tokenizer = MetaITokenizer::new(&config.tokenizer_path)?;
    let pad_id = tokenizer.pad_id().unwrap_or(0);

    // 2. Data
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

    // 3. Load Models
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();

    // Load Policy (Optimized)
    let policy_model = MetaIModel::new(&config.model, pad_id, &device);
    let policy_model = if let Some(epoch) = crate::train::train::find_latest_epoch(model_dir) {
        println!("Loading Policy model from epoch {}", epoch);
        let model_path = std::path::Path::new(model_dir)
            .join("checkpoint")
            .join(format!("model-{}.bin", epoch));
        policy_model
            .load_file(model_path, &recorder, &device)
            .expect("Failed to load policy weights")
    } else {
        println!("Warning: No checkpoint for Policy! DPO requires a converged SFT model.");
        policy_model
    };

    // Load Reference (Frozen)
    let ref_model = MetaIModel::new(&config.model, pad_id, &device);
    let ref_model = if let Some(epoch) = crate::train::train::find_latest_epoch(model_dir) {
        println!("Loading Reference model from epoch {}", epoch);
        let model_path = std::path::Path::new(model_dir)
            .join("checkpoint")
            .join(format!("model-{}.bin", epoch));
        ref_model
            .load_file(model_path, &recorder, &device)
            .expect("Failed to load ref weights")
    } else {
        ref_model
    };

    let model_wrapper = DPOWrapper::new(policy_model, ref_model);

    // 4. Learner
    let learner = LearnerBuilder::new(output_dir)
        .metric_train_numeric(LossMetric::<MyAutodiffBackend>::new())
        .metric_valid_numeric(LossMetric::<MyAutodiffBackend>::new())
        .with_file_checkpointer(recorder)
        .grads_accumulation(config.grads_accumulation)
        .num_epochs(config.num_epochs)
        .build(model_wrapper, config.optimizer.init(), config.learning_rate);

    // 5. Fit
    let _ = learner.fit(dataloader_train, dataloader_valid);

    Ok(())
}
