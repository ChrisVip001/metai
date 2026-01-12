use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Tensor};

use crate::data::data::TextBatch;
use crate::model::{MetaIConfig, MetaIModel};

use burn::train::metric::LossInput;
use burn::train::{TrainOutput, TrainStep, ValidStep};

#[derive(Clone, Debug)]
pub struct MetaITrainingConfig {
    pub model: MetaIConfig,
    pub optimizer: AdamWConfig,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub chinese_path: String,
    pub english_path: String,
    pub tokenizer_path: String,
    pub grads_accumulation: usize,
}

impl MetaITrainingConfig {
    pub fn new(model_config: MetaIConfig) -> Self {
        Self {
            model: model_config,
            optimizer: AdamWConfig::new(),
            batch_size: 32,
            num_epochs: 10,
            learning_rate: 1e-4,
            seed: 42,
            chinese_path: "4in1.txt".to_string(),
            english_path: "dataset.txt".to_string(),
            tokenizer_path: "tokenizer.json".to_string(),
            grads_accumulation: 1,
        }
    }
}

// 适配 Burn 0.19.1 的输出结构
#[derive(Clone)]
pub struct MetaIOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
}

#[derive(Clone, Debug)]
pub struct MetaIOutputSync {
    pub loss: f32,
}

impl<B: Backend> burn::train::metric::ItemLazy for MetaIOutput<B> {
    type ItemSync = MetaIOutputSync;

    fn sync(self) -> Self::ItemSync {
        MetaIOutputSync {
            loss: self.loss.into_scalar().elem::<f32>(),
        }
    }
}

impl burn::train::metric::ItemLazy for MetaIOutputSync {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

// 适配 LossMetric
impl<B: Backend> burn::train::metric::Adaptor<LossInput<B>> for MetaIOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> burn::train::metric::Adaptor<LossInput<B>> for MetaIOutputSync {
    fn adapt(&self) -> LossInput<B> {
        // 对于已同步的项，我们需要在一个（通常是 NdArray 或 CPU）后端重新构建它以满足 Adaptor 接口
        // 但通常 Metric 会在不同的阶段调用不同的 Adaptor
        LossInput::new(Tensor::from_floats([self.loss], &Default::default()))
    }
}

// 在 Burn 0.19.1 中实现 TrainStep
impl<B: AutodiffBackend> TrainStep<TextBatch<B>, MetaIOutput<B>> for MetaIModel<B> {
    fn step(&self, batch: TextBatch<B>) -> TrainOutput<MetaIOutput<B>> {
        let logits = self.forward(batch.inputs, None, None);
        let [batch_size, seq_len, vocab_size] = logits.dims();

        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_id as usize]))
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat);

        let grads = loss.backward();

        TrainOutput::new(self, grads, MetaIOutput { loss })
    }
}

impl<B: Backend> ValidStep<TextBatch<B>, MetaIOutput<B>> for MetaIModel<B> {
    fn step(&self, batch: TextBatch<B>) -> MetaIOutput<B> {
        let logits = self.forward(batch.inputs, None, None);
        let [batch_size, seq_len, vocab_size] = logits.dims();

        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_id as usize]))
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat);

        MetaIOutput { loss }
    }
}

use crate::data::sft::SFTBatch;

// SFT 训练步
impl<B: AutodiffBackend> TrainStep<SFTBatch<B>, MetaIOutput<B>> for MetaIModel<B> {
    fn step(&self, batch: SFTBatch<B>) -> TrainOutput<MetaIOutput<B>> {
        let logits = self.forward(batch.inputs, None, None);
        let [batch_size, seq_len, vocab_size] = logits.dims();

        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let mut targets_flat = batch.targets.reshape([batch_size * seq_len]);
        let mask_flat = batch.mask.reshape([batch_size * seq_len]);

        // 应用 Mask: 将 mask=0 的位置的 target 设为 pad_id
        // 这样 CrossEntropyLoss (配置了 ignore pad_id) 就会忽略这些位置
        // mask 是 Int, 需转为 Bool, mask=0 => bool=true (需要被覆盖)，反之 false
        // Wait: mask=1 is valid, mask=0 is ignore.
        // So we want to fill where mask == 0.
        let mask_bool = mask_flat.equal_elem(0);
        targets_flat = targets_flat.mask_fill(mask_bool, self.pad_id as i32);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_id as usize]))
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat);

        let grads = loss.backward();

        TrainOutput::new(self, grads, MetaIOutput { loss })
    }
}

impl<B: Backend> ValidStep<SFTBatch<B>, MetaIOutput<B>> for MetaIModel<B> {
    fn step(&self, batch: SFTBatch<B>) -> MetaIOutput<B> {
        let logits = self.forward(batch.inputs, None, None);
        let [batch_size, seq_len, vocab_size] = logits.dims();

        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let mut targets_flat = batch.targets.reshape([batch_size * seq_len]);
        let mask_flat = batch.mask.reshape([batch_size * seq_len]);

        let mask_bool = mask_flat.equal_elem(0);
        targets_flat = targets_flat.mask_fill(mask_bool, self.pad_id as i32);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_id as usize]))
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat);

        MetaIOutput { loss }
    }
}

pub mod distributed;
pub mod dpo;
pub mod grpo;
pub mod grpo_train_step;
pub mod reward;
pub mod sft;
pub mod train;

// --- Shared Utilities ---

pub fn find_latest_epoch(artifact_dir: &str) -> Option<usize> {
    let checkpoint_dir = std::path::Path::new(artifact_dir).join("checkpoint");
    if !checkpoint_dir.exists() {
        return None;
    }

    let mut max_epoch = 0;
    if let Ok(entries) = std::fs::read_dir(checkpoint_dir) {
        for entry in entries.flatten() {
            if let Some(file_name) = entry.file_name().to_str() {
                // 模型保存格式通常为 model-X.bin
                if file_name.starts_with("model-") && file_name.ends_with(".bin") {
                    if let Some(epoch_str) = file_name
                        .strip_prefix("model-")
                        .and_then(|s| s.strip_suffix(".bin"))
                    {
                        if let Ok(epoch) = epoch_str.parse::<usize>() {
                            if epoch > max_epoch {
                                max_epoch = epoch;
                            }
                        }
                    }
                }
            }
        }
    }

    if max_epoch > 0 {
        Some(max_epoch)
    } else {
        None
    }
}

pub fn load_model_checkpoint<B: Backend>(
    model: MetaIModel<B>,
    model_dir: &str,
    device: &B::Device,
) -> MetaIModel<B> {
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();

    if let Some(epoch) = find_latest_epoch(model_dir) {
        println!("Loading model checkpoint from epoch {}", epoch);
        let model_path = std::path::Path::new(model_dir)
            .join("checkpoint")
            .join(format!("model-{}.bin", epoch));

        match model.load_file(model_path, &recorder, device) {
            Ok(m) => m,
            Err(e) => {
                println!("Failed to load checkpoint: {}. Starting from scratch.", e);
                panic!("Failed to load checkpoint");
            }
        }
    } else {
        println!(
            "No checkpoint found at {}, utilizing initial weights.",
            model_dir
        );
        model
    }
}
