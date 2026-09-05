#![recursion_limit = "512"]
use burn::module::Module;
use clap::{Parser, Subcommand};
use metai::backend::{get_device, MyBackend};
use metai::infer::Generator;
use metai::train::train;
use metai::{MetaIConfig, MetaITokenizer};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 运行 30M 微型模型训练 (快速验证)
    TrainMicro {
        #[arg(short, long)]
        chinese_path: String,
        #[arg(short, long)]
        english_path: String,
        #[arg(short, long, default_value = "tokenizer.json")]
        tokenizer_path: String,
        #[arg(short, long, default_value = "output_micro")]
        output_dir: String,
    },
    /// 运行自定义配置模型训练
    TrainCustom {
        #[arg(short, long)]
        chinese_path: String,
        #[arg(short, long)]
        english_path: String,
        #[arg(short, long, default_value = "output_custom")]
        output_dir: String,
        #[arg(long, default_value_t = 32000)]
        vocab_size: usize,
        #[arg(long, default_value_t = 384)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 6)]
        num_layers: usize,
        #[arg(long, default_value_t = 6)]
        num_heads: usize,
        #[arg(long, default_value_t = 2)]
        num_kv_heads: usize,
        #[arg(long, default_value_t = 1024)]
        mlp_dim: usize,
        #[arg(long, default_value_t = 512)]
        max_seq_len: usize,
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
    },
    /// 运行 125M 中等模型单机训练
    TrainSmall {
        #[arg(short, long)]
        chinese_path: String,
        #[arg(short, long)]
        english_path: String,
        #[arg(short, long, default_value = "tokenizer.json")]
        tokenizer_path: String,
        #[arg(short, long, default_value = "output_small")]
        output_dir: String,
    },
    /// 运行双卡/多卡分布式训练
    TrainCluster {
        #[arg(short, long, default_value_t = 2)]
        world_size: usize,
        #[arg(short, long)]
        chinese_path: String,
        #[arg(short, long)]
        english_path: String,
    },
    /// 生成文本
    Generate {
        #[arg(short, long)]
        prompt: String,
        #[arg(short, long, default_value = "tokenizer.json")]
        tokenizer_path: String,
        #[arg(short, long, default_value_t = 50)]
        max_len: usize,
        #[arg(short, long, default_value = "/tmp/metai_local")]
        model_dir: String,
        #[arg(short, long, default_value_t = 0.8)]
        temperature: f32,
        #[arg(short, long, default_value_t = 40)]
        top_k: usize,
        #[arg(short, long, default_value_t = 0.9)]
        top_p: f32,
    },
    /// 投机解码生成（draft 模型 + target 模型）
    GenerateSpec {
        #[arg(short, long)]
        prompt: String,
        #[arg(short, long, default_value = "tokenizer.json")]
        tokenizer_path: String,
        #[arg(long, default_value_t = 50)]
        max_len: usize,
        #[arg(long, default_value = "/tmp/metai_local")]
        model_dir: String,
        #[arg(long, default_value = "/tmp/metai_local_quant")]
        draft_dir: String,
        #[arg(long, default_value_t = 4)]
        lookahead: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
    },
    /// 运行 SFT 指令微调
    TrainSft {
        #[arg(short, long)]
        data_path: String,
        #[arg(short, long)]
        model_dir: String,
        #[arg(short, long, default_value = "/tmp/metai_sft")]
        output_dir: String,
    },
    /// 运行 DPO 偏好对齐
    TrainDpo {
        #[arg(short, long)]
        data_path: String,
        #[arg(short, long)]
        model_dir: String,
        #[arg(short, long, default_value = "/tmp/metai_dpo")]
        output_dir: String,
    },
    /// 运行 GRPO 强化学习（真实 rollout + 规则奖励）
    TrainGrpo {
        #[arg(short, long)]
        data_path: String,
        #[arg(short, long)]
        model_dir: String,
        #[arg(short, long, default_value = "/tmp/metai_grpo")]
        output_dir: String,
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer_path: String,
        #[arg(long, default_value_t = 1)]
        num_epochs: usize,
        #[arg(long, default_value_t = 5e-7)]
        learning_rate: f64,
        #[arg(long, default_value_t = 2)]
        batch_size: usize,
        #[arg(long, default_value_t = 4)]
        group_size: usize,
        #[arg(long, default_value_t = 96)]
        max_new_tokens: usize,
        #[arg(long, value_enum, default_value = "rule")]
        reward: metai::train::reward::RewardKind,
        #[arg(long, default_value_t = 0.9)]
        temperature: f32,
        #[arg(long, default_value_t = 50)]
        top_k: usize,
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
    },
    /// 对模型进行 INT4 量化
    Quantize {
        #[arg(short, long)]
        input_path: String,
        #[arg(short, long)]
        output_path: String,
        #[arg(short, long, default_value = "tokenizer.json")]
        tokenizer_path: String,
    },
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::TrainMicro {
            chinese_path,
            english_path,
            tokenizer_path: _,
            output_dir,
        } => {
            println!("Starting MetaI Nano (6.6M) Training...");
            train::run_micro_training(&chinese_path, &english_path, &output_dir)?;
        }
        Commands::TrainSmall {
            chinese_path,
            english_path,
            tokenizer_path: _,
            output_dir,
        } => {
            println!("Starting MetaI Small (125M) Training...");
            train::run_small_training(&chinese_path, &english_path, &output_dir)?;
        }
        Commands::TrainCustom {
            chinese_path,
            english_path,
            output_dir,
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            mlp_dim,
            max_seq_len,
            batch_size,
        } => {
            println!("Starting Custom Model Training...");

            let config = MetaIConfig {
                vocab_size,
                hidden_dim,
                num_layers,
                num_heads,
                num_kv_heads,
                mlp_dim,
                max_seq_len,
                dropout: 0.0,
                num_experts: 4,
                active_experts: 1,
                num_shared_experts: 0,
            };

            // Call run_custom_training logic (needs to be added or inline here)
            // For simplicity, we can reuse logic or create a helper
            let training_config = metai::train::MetaITrainingConfig {
                chinese_path: chinese_path.to_string(),
                english_path: english_path.to_string(),
                tokenizer_path: "tokenizer.json".to_string(),
                model: config,
                optimizer: burn::optim::AdamWConfig::new(),
                batch_size,
                num_epochs: 10,
                learning_rate: 1e-3,
                seed: 42,
                grads_accumulation: 4,
            };
            metai::train::train::train::<metai::backend::MyAutodiffBackend>(
                &output_dir,
                training_config,
                get_device(),
            )?;
        }
        Commands::TrainCluster {
            world_size,
            chinese_path,
            english_path,
        } => {
            println!(
                "Starting Multi-GPU Cluster Training (World Size: {})...",
                world_size
            );
            // 集群训练直接上 DeepSeek-V3.2 顶配规模 (685B MoE + 128K Context)
            let mut config =
                metai::train::MetaITrainingConfig::new(MetaIConfig::deepseek_v3_2_685b());
            config.chinese_path = chinese_path;
            config.english_path = english_path;

            // 针对 2026 年 SOTA 规模集群调优
            config.batch_size = 1; // 685B 极其吃显存，单卡物理 Batch 设为 1
            config.grads_accumulation = 256; // 极大累积以榨干 128K 上下文的吞吐潜力，并对抗网络延迟
            config.learning_rate = 8e-5; // 更大的规模需要更保守的学习率

            metai::train::distributed::DistributedTrainer::launch_cluster(world_size, config)?;
        }
        Commands::Generate {
            prompt,
            tokenizer_path,
            max_len,
            model_dir,
            temperature,
            top_k,
            top_p,
        } => {
            let tokenizer = MetaITokenizer::new(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
            let config = MetaIConfig::small();
            let device = get_device();
            let generator =
                Generator::<MyBackend>::from_checkpoint(&model_dir, config, tokenizer, &device)?;
            println!("生成中...");
            let output = generator.generate(&prompt, max_len, temperature, top_k, top_p);
            println!("\n=== 生成结果 ===\n{}", output);
        }
        Commands::TrainSft {
            data_path,
            model_dir,
            output_dir,
        } => {
            metai::train::sft::run_sft_training(&data_path, &model_dir, &output_dir)?;
        }
        Commands::TrainDpo {
            data_path,
            model_dir,
            output_dir,
        } => {
            metai::train::dpo::run_dpo_training(&data_path, &model_dir, &output_dir)?;
        }
        Commands::TrainGrpo {
            data_path,
            model_dir,
            output_dir,
            tokenizer_path,
            num_epochs,
            learning_rate,
            batch_size,
            group_size,
            max_new_tokens,
            reward,
            temperature,
            top_k,
            top_p,
        } => {
            println!("Starting GRPO RL Training...");
            let mut tcfg = metai::train::MetaITrainingConfig::new(MetaIConfig::small());
            tcfg.tokenizer_path = tokenizer_path;
            tcfg.num_epochs = num_epochs;
            tcfg.learning_rate = learning_rate;
            tcfg.batch_size = batch_size;
            let grpo_cfg = metai::train::grpo::GRPOConfig {
                beta: 0.1,
                clip_eps: 0.2,
                group_size,
                max_new_tokens,
                reward,
                temperature,
                top_k,
                top_p,
            };
            metai::train::grpo_train_step::run_grpo_training(
                &data_path,
                &model_dir,
                &output_dir,
                &tcfg,
                &grpo_cfg,
            )?;
        }
        Commands::GenerateSpec {
            prompt,
            tokenizer_path,
            max_len,
            model_dir,
            draft_dir,
            lookahead,
            temperature,
        } => {
            let tokenizer = MetaITokenizer::new(&tokenizer_path)?;
            let config = MetaIConfig::small();
            let device = get_device();
            let pad_id = tokenizer.pad_id().unwrap_or(0);
            let target = metai::infer::load_model_from_checkpoint::<MyBackend>(
                &model_dir,
                config.clone(),
                pad_id,
                &device,
            )?;
            let draft = metai::infer::load_model_from_checkpoint::<MyBackend>(
                &draft_dir,
                config,
                pad_id,
                &device,
            )?;
            let generator = metai::infer::SpeculativeGenerator::new(draft, target, tokenizer);
            println!("投机解码生成中 (lookahead={})...", lookahead);
            let output = generator.generate(&prompt, max_len, lookahead, temperature);
            println!("\n=== 生成结果 ===\n{}", output);
        }
        Commands::Quantize {
            input_path,
            output_path,
            tokenizer_path,
        } => {
            // 量化逻辑... (简化实现以保持响应长度)
            println!(
                "Starting Quantization from {} to {}...",
                input_path, output_path
            );
            let tokenizer = MetaITokenizer::new(&tokenizer_path)?;
            let config = MetaIConfig::small();
            let device = get_device();
            use burn::record::{BinFileRecorder, FullPrecisionSettings};
            let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
            let model: metai::MetaIModel<MyBackend> =
                metai::MetaIModel::new(&config, tokenizer.pad_id().unwrap_or(0), &device);
            // 取最新 epoch（替代硬编码 model-1.bin）
            let epoch = metai::train::find_latest_epoch(&input_path)
                .ok_or_else(|| anyhow::anyhow!("No checkpoint found in {}", input_path))?;
            let checkpoint_path = std::path::Path::new(&input_path).join("checkpoint");
            let model = model.load_file(
                checkpoint_path.join(format!("model-{}.bin", epoch)),
                &recorder,
                &device,
            )?;
            let quantized = model.quantize_int4(&device);
            std::fs::create_dir_all(&output_path)?;
            quantized.save_file(
                std::path::Path::new(&output_path).join("quantized_model.bin"),
                &recorder,
            )?;
            println!("Quantization finished.");
        }
    }
    Ok(())
}
