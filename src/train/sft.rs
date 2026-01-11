use crate::backend::{get_device, MyAutodiffBackend, MyBackend};
use burn::data::dataloader::DataLoaderBuilder;

use burn::module::Module;
use burn::train::metric::LossMetric;
use burn::train::LearnerBuilder;

use crate::data::sft::{InstructionDataset, SFTBatcher};
use crate::data::MetaITokenizer;
use crate::model::MetaIModel;
use crate::train::MetaITrainingConfig;

pub fn run_sft_training(
    data_path: &str,
    model_dir: &str, // Load pre-trained model from here
    output_dir: &str,
) -> anyhow::Result<()> {
    let device = get_device();

    // 1. Load Config & Tokenizer
    // Assume config is in model_dir or use default SFT config
    // For simplicity, we use a default SFT config based on 'small' but with lower LR
    let mut config = MetaITrainingConfig::new(crate::model::MetaIConfig::small());
    config.learning_rate = 1e-5; // Lower LR for Fine-tuning
    config.num_epochs = 3;
    config.batch_size = 4; // Adjust based on memory
    config.tokenizer_path = "tokenizer.json".to_string();

    let tokenizer = MetaITokenizer::new(&config.tokenizer_path)?;
    let pad_id = tokenizer.pad_id().unwrap_or(0);

    // 2. Prepare Dataset (SFT)
    // SFT dataset is usually smaller, so we might not need Valid set strictly if data is scarce,
    // but good practice to have one. Here we use same file for simplicity or split it.
    // For this demo, we assume data_path contains JSONL.
    let dataset = InstructionDataset::from_file(data_path, &tokenizer, config.model.max_seq_len)?;

    // Split 90/10
    let total_len = burn::data::dataset::Dataset::len(&dataset);
    let _train_len = (total_len as f32 * 0.9) as usize;
    // Burn's Dataset doesn't have easy split? We can use transform or just load two files.
    // Let's just use the full dataset for training for now.

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
        .build(dataset); // use same for valid in this simplified demo

    // 3. Load Pre-trained Model
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();

    // Construct empty model first
    let model = MetaIModel::new(&config.model, pad_id, &device);

    // Find latest checkpoint in model_dir
    let model = if let Some(epoch) = crate::train::train::find_latest_epoch(model_dir) {
        println!("Loading pre-trained model from epoch {}", epoch);
        let model_path = std::path::Path::new(model_dir)
            .join("checkpoint")
            .join(format!("model-{}.bin", epoch));
        model
            .load_file(model_path, &recorder, &device)
            .expect("Failed to load weights")
    } else {
        println!(
            "Warning: No pre-trained checkpoint found at {}, starting from scratch!",
            model_dir
        );
        model
    };

    // 4. Build Learner
    let learner = LearnerBuilder::new(output_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(recorder)
        .grads_accumulation(config.grads_accumulation)
        .num_epochs(config.num_epochs)
        .build(model, config.optimizer.init(), config.learning_rate);

    // 5. Fit
    let _ = learner.fit(dataloader_train, dataloader_valid);

    Ok(())
}
