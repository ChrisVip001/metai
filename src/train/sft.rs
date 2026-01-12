use crate::backend::{get_device, MyAutodiffBackend, MyBackend};
use burn::data::dataloader::DataLoaderBuilder;

use burn::train::metric::LossMetric;
use burn::train::LearnerBuilder;

use crate::data::sft::{InstructionDataset, SFTBatcher};
use crate::data::MetaITokenizer;
use crate::model::MetaIModel;
use crate::train::MetaITrainingConfig;

pub fn run_sft_training(
    data_path: &str,
    model_dir: &str, // 从此处加载预训练模型
    output_dir: &str,
) -> anyhow::Result<()> {
    let device = get_device();

    // 1. 加载配置与分词器
    // 假设配置在 model_dir 中，或者使用默认的 SFT 配置
    // 为简单起见，我们基于 'small' 配置使用默认的 SFT 配置，但降低学习率
    let mut config = MetaITrainingConfig::new(crate::model::MetaIConfig::small());
    config.learning_rate = 1e-5; // 微调时使用较低的学习率
    config.num_epochs = 3;
    config.batch_size = 4; // 根据显存大小调整
    config.tokenizer_path = "tokenizer.json".to_string();

    let tokenizer = MetaITokenizer::new(&config.tokenizer_path)?;
    let pad_id = tokenizer.pad_id().unwrap_or(0);

    // 2. 准备数据集 (SFT)
    // SFT 数据集通常较小，如果数据稀缺，可能不需要严格的验证集，
    // 但拥有验证集是一个好习惯。这里为简单起见使用相同的文件，或者将其拆分。
    // 在本演示中，我们假设 data_path 包含 JSONL 文件。
    let dataset = InstructionDataset::from_file(data_path, &tokenizer, config.model.max_seq_len)?;

    // 90/10 拆分
    let total_len = burn::data::dataset::Dataset::len(&dataset);
    let _train_len = (total_len as f32 * 0.9) as usize;
    // Burn 的 Dataset 没有简单的拆分方法？我们可以使用 transform 或者直接加载两个文件。
    // 目前我们先使用整个数据集进行训练。

    let batcher = SFTBatcher::<MyAutodiffBackend>::new(device.clone(), pad_id);
    let batcher_valid = SFTBatcher::<MyBackend>::new(device.clone(), pad_id);

    let dataloader_train = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(4)
        .build(dataset.clone());

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(4)
        .build(dataset); // 在此简化的演示中，验证集使用相同的数据

    // 3. 加载预训练模型
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();

    // 先构建一个空模型
    let model = MetaIModel::new(&config.model, pad_id, &device);
    let model = crate::train::load_model_checkpoint(model, model_dir, &device);

    // 4. 构建 Learner
    let learner = LearnerBuilder::new(output_dir)
        .metric_train_numeric(LossMetric::<MyAutodiffBackend>::new())
        .metric_valid_numeric(LossMetric::<MyAutodiffBackend>::new())
        .with_file_checkpointer(recorder)
        .grads_accumulation(config.grads_accumulation)
        .num_epochs(config.num_epochs)
        .build(model, config.optimizer.init(), config.learning_rate);

    // 5. 开始拟合
    let _ = learner.fit(dataloader_train, dataloader_valid);

    Ok(())
}
