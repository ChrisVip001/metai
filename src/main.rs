use clap::{Parser, Subcommand};
use metai::train::train;
use metai::infer::Generator;
use metai::{MetaIConfig, MetaITokenizer};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::module::Module;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 运行 1M 小模型训练测试
    TrainTiny {
        #[arg(short, long, default_value = "4in1.txt")]
        chinese_path: String,
        #[arg(short, long, default_value = "dataset.txt")]
        english_path: String,
        #[arg(short, long, default_value = "tokenizer.json")]
        tokenizer_path: String,
    },
    /// 运行本地优化版训练 (适合 16GB 内存设备)
    TrainLocal {
        #[arg(short, long, default_value = "dataset_local_zh.txt")]
        chinese_path: String,
        #[arg(short, long, default_value = "dataset_local_en.txt")]
        english_path: String,
        #[arg(short, long, default_value = "tokenizer.json")]
        tokenizer_path: String,
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
    /// 对模型进行 INT4 量化压缩
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
        Commands::TrainTiny {
            chinese_path,
            english_path,
            tokenizer_path: _,
        } => {
            println!("Starting MetaI Tiny Training...");
            train::run_tiny_test(&chinese_path, &english_path)?;
        }
        Commands::TrainLocal {
            chinese_path,
            english_path,
            tokenizer_path: _,
        } => {
            println!("Starting MetaI Local Optimization Training...");
            train::run_local_training(&chinese_path, &english_path)?;
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
            println!("正在加载模型和分词器...");
            
            // 加载分词器
            let tokenizer = MetaITokenizer::new(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
            
            // 确定模型配置（根据检查点目录推断，或使用默认配置）
            let config = if model_dir.contains("tiny") {
                MetaIConfig::tiny()
            } else {
                MetaIConfig::local_optimized()
            };
            
            // 创建设备
            let device = WgpuDevice::default();
            
            // 加载模型
            let generator = Generator::<Wgpu>::from_checkpoint(
                &model_dir,
                config,
                tokenizer,
                &device,
            )?;
            
            println!("生成中...");
            let output = generator.generate(&prompt, max_len, temperature, top_k, top_p);
            
            println!("\n=== 生成结果 ===");
            println!("{}", output);
        }
        Commands::Quantize {
            input_path,
            output_path,
            tokenizer_path,
        } => {
            println!("正在加载模型进行量化...");
            
            // 加载分词器以获取配置
            let tokenizer = MetaITokenizer::new(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
            
            // 确定模型配置
            let config = if input_path.contains("tiny") {
                MetaIConfig::tiny()
            } else {
                MetaIConfig::local_optimized()
            };
            
            let device = WgpuDevice::default();
            let pad_id = tokenizer.pad_id().unwrap_or(0);
            
            // 加载原始模型
            use burn::record::{BinFileRecorder, FullPrecisionSettings};
            use std::path::Path;
            
            let checkpoint_dir = Path::new(&input_path).join("checkpoint");
            let mut max_epoch = 0;
            if let Ok(entries) = std::fs::read_dir(&checkpoint_dir) {
                for entry in entries.flatten() {
                    if let Some(file_name) = entry.file_name().to_str() {
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
            
            if max_epoch == 0 {
                anyhow::bail!("No checkpoint found in {:?}", checkpoint_dir);
            }
            
            let model: metai::MetaIModel<Wgpu> = metai::MetaIModel::new(&config, pad_id, &device);
            let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
            let model_path = checkpoint_dir.join(format!("model-{}.bin", max_epoch));
            
            let model = model.load_file(&model_path, &recorder, &device)
                .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
            
            println!("正在执行 INT4 量化...");
            let quantized_model = model.quantize_int4(&device);
            
            // 保存量化后的模型
            std::fs::create_dir_all(&output_path)?;
            let output_checkpoint = Path::new(&output_path).join("checkpoint");
            std::fs::create_dir_all(&output_checkpoint)?;
            let output_model_path = output_checkpoint.join(format!("model-{}-quantized.bin", max_epoch));
            
            quantized_model.save_file(&output_model_path, &recorder)
                .map_err(|e| anyhow::anyhow!("Failed to save quantized model: {}", e))?;
            
            println!("量化完成！模型已保存到: {:?}", output_model_path);
        }
    }

    Ok(())
}
