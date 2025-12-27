use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;

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

// 在 0.19 中，最稳妥的做法是让 ItemSync 保持与原 Backend 一致（或使用 Simple backend）
// 避免过度深入 InnerBackend 的 Trait 迷宫
impl<B: Backend> burn::train::metric::ItemLazy for MetaIOutput<B> {
    type ItemSync = MetaIOutput<B>;

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

pub mod train;
