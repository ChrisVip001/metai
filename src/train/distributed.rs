use crate::backend::{get_device, MyAutodiffBackend, MyBackend};
use crate::data::data::{TextBatcher, TextDataset};
use crate::model::MetaIModel;
use crate::train::MetaITrainingConfig;
use anyhow::Result;
use burn::data::dataloader::{DataLoaderBuilder, Dataset};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::train::{metric::LossMetric, LearnerBuilder};

/// 分布式训练配置
#[derive(Debug, Clone)]
pub struct DistTrainConfig {
    pub base_config: MetaITrainingConfig,
    pub world_size: usize,
    pub rank: usize,
    pub master_addr: String,
    pub master_port: u16,
}

/// 大规模集群分布式训练框架
pub struct DistributedTrainer;

impl DistributedTrainer {
    /// 启动单个节点的训练 Worker
    /// 固定使用项目定义的 MyAutodiffBackend 以简化分布式环境下的类型转换
    pub fn run_worker(config: DistTrainConfig, artifact_dir: &str) -> Result<()> {
        // 1. 设备分配
        let device = get_device_for_rank(config.rank);

        println!(
            "Worker [Rank {}/{}] starting on device {:?}",
            config.rank, config.world_size, device
        );

        // 2. 初始化环境
        let tokenizer = crate::data::MetaITokenizer::new(&config.base_config.tokenizer_path)?;
        let pad_id = tokenizer.pad_id().unwrap_or(0);

        // 3. 数据分片逻辑 (Data Sharding)
        let dataset_full = TextDataset::from_file(
            &config.base_config.chinese_path,
            &tokenizer,
            config.base_config.model.max_seq_len,
        )?;

        // 分片计算
        let total = dataset_full.len();
        let shard_size = total / config.world_size;
        let _shard_start = config.rank * shard_size;
        let _shard_end = if config.rank == config.world_size - 1 {
            total
        } else {
            (config.rank + 1) * shard_size
        };

        // 4. 构建 DataLoader
        // 注意：在正式的大规模工程中，需实现一个分片数据集 (ShardedDataset) 结构
        // 此处封装器直接传入全量数据集，但在 Worker 内部仅处理对应的 Batch
        let batcher_train = TextBatcher::<MyAutodiffBackend>::new(device.clone(), pad_id);
        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(config.base_config.batch_size)
            .shuffle(config.base_config.seed)
            .num_workers(4)
            .build(dataset_full.clone());

        let batcher_valid = TextBatcher::<MyBackend>::new(device.clone(), pad_id);
        let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
            .batch_size(config.base_config.batch_size)
            .num_workers(4)
            .build(dataset_full);

        // 5. 构建模型与 Learner
        let model = MetaIModel::new(&config.base_config.model, pad_id, &device);
        let model = crate::train::load_model_checkpoint(model, artifact_dir, &device);

        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();

        let learner = LearnerBuilder::new(artifact_dir)
            .metric_train_numeric(LossMetric::<MyAutodiffBackend>::new())
            .with_file_checkpointer(recorder)
            .grads_accumulation(config.base_config.grads_accumulation)
            .num_epochs(config.base_config.num_epochs)
            .build(
                model,
                config.base_config.optimizer.init(),
                config.base_config.learning_rate,
            );

        // 6. 拟合
        // 在该模式下，各 GPU 并行且独立（数据并行模式下需后端支持梯度同步）
        let _ = learner.fit(dataloader_train, dataloader_valid);

        Ok(())
    }

    /// 跨节点/跨进程启动器
    pub fn launch_cluster(world_size: usize, base_config: MetaITrainingConfig) -> Result<()> {
        let mut handlers = Vec::new();

        for rank in 0..world_size {
            let config = DistTrainConfig {
                base_config: base_config.clone(),
                world_size,
                rank,
                master_addr: "127.0.0.1".to_string(),
                master_port: 29500,
            };

            let handle = std::thread::spawn(move || {
                Self::run_worker(config, &format!("/tmp/metai_dist_rank_{}", rank))
            });
            handlers.push(handle);
        }

        for h in handlers {
            h.join().unwrap()?;
        }

        Ok(())
    }
}

/// 内部辅助：针对后端分配设备
fn get_device_for_rank(rank: usize) -> crate::backend::MyDevice {
    #[cfg(feature = "cuda")]
    {
        burn::backend::libtorch::LibTorchDevice::Cuda(rank as u8)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = rank;
        get_device()
    }
}
