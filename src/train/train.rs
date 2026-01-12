use crate::backend::{get_device, MyAutodiffBackend};
use burn::data::dataloader::DataLoaderBuilder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::LearnerBuilder;

use crate::data::data::{TextBatcher, TextDataset};
use crate::data::MetaITokenizer;
use crate::model::config::MetaIConfig;
use crate::model::MetaIModel;
use crate::train::MetaITrainingConfig;

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: MetaITrainingConfig,
    device: B::Device,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(artifact_dir)?;

    // 0. Tokenizer
    let tokenizer_path = std::path::Path::new(&config.tokenizer_path);
    if !tokenizer_path.exists() {
        println!(
            "Tokenizer not found at {:?}, training a new one...",
            tokenizer_path
        );
        MetaITokenizer::train(
            &vec![&config.chinese_path, &config.english_path],
            &config.tokenizer_path,
            config.model.vocab_size,
        )?;
    }
    let tokenizer = MetaITokenizer::new(&config.tokenizer_path)?;
    let pad_id = tokenizer.pad_id().unwrap_or(0);

    // 1. 数据准备
    let dataset_train =
        TextDataset::from_file(&config.chinese_path, &tokenizer, config.model.max_seq_len)?;
    let dataset_valid =
        TextDataset::from_file(&config.english_path, &tokenizer, config.model.max_seq_len)?;

    let batcher_train = TextBatcher::<B>::new(device.clone(), pad_id);
    let batcher_valid = TextBatcher::<B::InnerBackend>::new(device.clone(), pad_id);

    // 2. 创建数据加载器
    let dataloader_train = DataLoaderBuilder::<B, _, _>::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(4)
        .build(dataset_train);

    let dataloader_valid = DataLoaderBuilder::<B::InnerBackend, _, _>::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(4)
        .build(dataset_valid);

    // 3. 构建模型
    let model = MetaIModel::new(&config.model, pad_id, &device);

    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();

    // 4. 构建 Learner
    let learner_builder = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::<B>::new())
        .metric_valid_numeric(LossMetric::<B>::new())
        .with_file_checkpointer(recorder)
        .grads_accumulation(config.grads_accumulation)
        .num_epochs(config.num_epochs)
        .summary();

    let learner = if let Some(epoch) = crate::train::find_latest_epoch(artifact_dir) {
        println!("Found checkpoint at epoch {}, resuming training...", epoch);
        learner_builder.checkpoint(epoch).build(
            model,
            config.optimizer.init(),
            config.learning_rate,
        )
    } else {
        learner_builder.build(model, config.optimizer.init(), config.learning_rate)
    };

    // 5. 开始拟合
    let _ = learner.fit(dataloader_train, dataloader_valid);

    Ok(())
}

pub fn run_small_training(
    chinese_path: &str,
    english_path: &str,
    output_dir: &str,
) -> anyhow::Result<()> {
    let device = get_device();
    let config = MetaITrainingConfig {
        chinese_path: chinese_path.to_string(),
        english_path: english_path.to_string(),
        tokenizer_path: "tokenizer.json".to_string(),
        model: MetaIConfig::small(),
        optimizer: burn::optim::AdamWConfig::new(),
        batch_size: 2, // 125M 模型物理 Batch 较小，依赖累积
        num_epochs: 20,
        learning_rate: 2e-4,
        seed: 42,
        grads_accumulation: 16, // 等效 Batch=32
    };

    train::<MyAutodiffBackend>(output_dir, config, device)?;
    Ok(())
}
