use anyhow::{ensure, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use metai::{
    backend::get_device,
    infer::SpeculativeGenerator,
    train::{self, Checkpoint, Schedule},
    Generator, MetaIConfig,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "MetaI 全链路大模型研发框架")]
struct Cli {
    /// Select a compute runtime compiled into this binary.
    #[arg(long, value_enum, global = true)]
    device: Option<metai::backend::Runtime>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum Size {
    Tiny,
    Micro,
    Small,
    #[value(name = "125m")]
    Prototype125m,
}

impl Size {
    fn config(self) -> MetaIConfig {
        match self {
            Self::Tiny => MetaIConfig::tiny(),
            Self::Micro => MetaIConfig::micro(),
            Self::Small => MetaIConfig::small(),
            Self::Prototype125m => MetaIConfig::prototype_125m(),
        }
    }
}

#[derive(Args)]
struct Corpus {
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,
    #[arg(long, alias = "chinese-path")]
    train_path: String,
    #[arg(long, alias = "english-path")]
    valid_path: String,
    #[arg(long)]
    tokenizer_path: Option<PathBuf>,
    #[arg(long)]
    output_dir: PathBuf,
    #[command(flatten)]
    schedule: Schedule,
}

#[derive(Args)]
struct Post {
    #[arg(long)]
    data_path: String,
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[command(flatten)]
    schedule: Schedule,
}

#[derive(Subcommand)]
enum Command {
    /// Estimate packed weights and batch capacity without creating tensors or starting workers.
    PlanCluster(train::distributed::CapacityPlan),
    /// Fit a tokenizer on training text and train a causal language model.
    Pretrain {
        #[arg(long, value_enum, default_value = "tiny")]
        size: Size,
        #[arg(long, conflicts_with = "size")]
        config: Option<PathBuf>,
        #[command(flatten)]
        corpus: Corpus,
    },
    /// Synchronous data parallel training on local devices (Flex ranks can share a CPU).
    TrainCluster {
        #[arg(long, default_value_t = 2)]
        world_size: usize,
        #[arg(long, default_value_t = 1)]
        accumulation_steps: usize,
        #[arg(long, value_enum, default_value = "125m")]
        size: Size,
        #[arg(long, conflicts_with = "size")]
        config: Option<PathBuf>,
        /// Repeat for multiple UTF-8 line-oriented training files.
        #[arg(long, conflicts_with_all=["chinese_path", "english_path"])]
        train_path: Vec<PathBuf>,
        #[arg(long, requires = "english_path")]
        chinese_path: Option<PathBuf>,
        #[arg(long, requires = "chinese_path")]
        english_path: Option<PathBuf>,
        #[arg(long)]
        valid_path: Option<PathBuf>,
        #[arg(long)]
        tokenizer_path: Option<PathBuf>,
        #[arg(long, default_value = "output_cluster")]
        output_dir: PathBuf,
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        #[command(flatten)]
        schedule: Schedule,
    },
    ResumePretrain {
        /// Change data-parallel topology while preserving the original global batch.
        #[arg(long, requires = "batch_size")]
        world_size: Option<usize>,
        #[arg(long, requires = "world_size")]
        batch_size: Option<usize>,
        #[arg(long, requires = "world_size")]
        accumulation_steps: Option<usize>,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, required = true)]
        train_path: Vec<PathBuf>,
        #[arg(long)]
        valid_path: Option<PathBuf>,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        additional_epochs: usize,
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
    },
    TrainMicro(Corpus),
    TrainSmall(Corpus),
    TrainCustom {
        #[command(flatten)]
        corpus: Corpus,
        #[arg(long, default_value_t = 320)]
        vocab_size: usize,
        #[arg(long, default_value_t = 16)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 1)]
        num_layers: usize,
        #[arg(long, default_value_t = 2)]
        num_heads: usize,
        #[arg(long, default_value_t = 1)]
        num_kv_heads: usize,
        #[arg(long, default_value_t = 32)]
        mlp_dim: usize,
        #[arg(long, default_value_t = 96)]
        max_seq_len: usize,
    },
    Lab {
        #[arg(value_enum)]
        kind: metai::labs::Kind,
        #[arg(long)]
        output_dir: PathBuf,
        #[command(flatten)]
        schedule: Schedule,
    },
    PlayAgent {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        case: usize,
        #[arg(long)]
        imitation: bool,
    },
    TrainAgent {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[command(flatten)]
        schedule: Schedule,
    },
    TrainDecision {
        #[command(flatten)]
        post: Post,
        #[arg(long)]
        calibration_path: PathBuf,
        #[arg(long, value_enum, default_value = "last")]
        pooling: metai::decision::Pooling,
        #[arg(long)]
        test_path: PathBuf,
    },
    Decide {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        text: String,
    },
    TrainThinking {
        #[command(flatten)]
        post: Post,
        #[arg(long, default_value_t = 4)]
        budget: usize,
        #[arg(long, default_value_t = 0.03)]
        cost: f64,
    },
    Think {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 8)]
        max_len: usize,
        #[arg(long, default_value_t = 0.8)]
        threshold: f64,
        #[arg(long)]
        budget: Option<usize>,
    },
    TrainMultimodal {
        #[command(flatten)]
        post: Post,
        #[arg(long)]
        audio: bool,
        /// Learn a vocabulary readout conditioned on both the sensory prefix and text.
        #[arg(long)]
        train_readout: bool,
        /// Tokenize the answer as a continuation of the rendered prompt.
        #[arg(long)]
        continuation_targets: bool,
    },
    GenerateMultimodal {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        data_path: PathBuf,
        #[arg(long, default_value_t = 4)]
        max_len: usize,
    },
    TrainSft(Post),
    TrainDpo {
        #[command(flatten)]
        post: Post,
        #[arg(long, default_value_t = 0.1)]
        beta: f64,
    },
    TrainLora {
        #[command(flatten)]
        post: Post,
        #[arg(long, default_value_t = 4)]
        rank: usize,
        #[arg(long, default_value_t = 8.)]
        alpha: f64,
    },
    Distill {
        #[command(flatten)]
        post: Post,
        #[arg(long, default_value_t = 2.)]
        temperature: f64,
    },
    TrainGrpo {
        #[command(flatten)]
        post: Post,
        #[command(flatten)]
        rollout: train::grpo_train_step::RolloutConfig,
    },
    TrainGspo {
        #[command(flatten)]
        post: Post,
        #[command(flatten)]
        rollout: train::grpo_train_step::RolloutConfig,
    },
    TrainPpo {
        #[command(flatten)]
        post: Post,
        #[command(flatten)]
        rollout: train::grpo_train_step::RolloutConfig,
    },
    TrainCispo {
        #[command(flatten)]
        post: Post,
        #[command(flatten)]
        rollout: train::grpo_train_step::RolloutConfig,
    },
    Generate {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        chat: bool,
        #[arg(long, default_value_t = 16)]
        max_len: usize,
        #[arg(long, default_value_t = 0.)]
        temperature: f32,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
    },
    /// Answer JSONL chat questions with greedy decoding; emit raw completions as JSONL.
    EvalChat {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        questions: PathBuf,
        #[arg(long, default_value_t = 32)]
        max_len: usize,
    },
    GenerateSpec {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        draft_dir: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 16)]
        max_len: usize,
        #[arg(long, default_value_t = 4)]
        lookahead: usize,
    },
    Quantize {
        #[arg(long)]
        input_path: PathBuf,
        #[arg(long)]
        output_path: PathBuf,
    },
}

fn pretrain(c: Corpus, model: MetaIConfig) -> Result<()> {
    train::train::pretrain(
        model,
        &c.train_path,
        &c.valid_path,
        c.tokenizer_path.as_deref(),
        &c.output_dir,
        &c.schedule,
        c.checkpoint_dir.as_deref(),
    )
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(runtime) = cli.device {
        metai::backend::select(runtime)?;
    }
    match cli.command {
        Command::PlanCluster(plan) => {
            println!("{}", serde_json::to_string_pretty(&plan.report()?)?);
            Ok(())
        }
        Command::Pretrain {
            config: Some(path),
            corpus,
            ..
        } => pretrain(corpus, serde_json::from_reader(std::fs::File::open(path)?)?),
        Command::Pretrain { size, corpus, .. } => pretrain(corpus, size.config()),
        Command::TrainCluster {
            world_size,
            accumulation_steps,
            size,
            config,
            mut train_path,
            chinese_path,
            english_path,
            valid_path,
            tokenizer_path,
            output_dir,
            checkpoint_dir,
            schedule,
        } => {
            train_path.extend(chinese_path);
            train_path.extend(english_path);
            let model = match config {
                Some(path) => serde_json::from_reader(std::fs::File::open(path)?)?,
                None => size.config(),
            };
            train::train::pretrain_sources(
                model,
                train::train::Sources {
                    train: &train_path.iter().map(|p| p.as_path()).collect::<Vec<_>>(),
                    valid: valid_path.as_deref(),
                },
                tokenizer_path.as_deref(),
                &output_dir,
                &schedule,
                checkpoint_dir.as_deref(),
                Some(train::distributed::Topology {
                    world_size,
                    accumulation_steps,
                }),
            )
        }
        Command::ResumePretrain {
            world_size,
            batch_size,
            accumulation_steps,
            model_dir,
            train_path,
            valid_path,
            output_dir,
            additional_epochs,
            checkpoint_dir,
        } => train::train::resume_with_topology(
            &model_dir,
            train::train::Sources {
                train: &train_path.iter().map(|p| p.as_path()).collect::<Vec<_>>(),
                valid: valid_path.as_deref(),
            },
            &output_dir,
            additional_epochs,
            checkpoint_dir.as_deref(),
            world_size.zip(batch_size).map(|(world_size, micro)| {
                (
                    train::distributed::Topology {
                        world_size,
                        accumulation_steps: accumulation_steps.unwrap_or(1),
                    },
                    micro,
                )
            }),
        ),
        Command::TrainMicro(c) => pretrain(c, MetaIConfig::micro()),
        Command::TrainSmall(c) => pretrain(c, MetaIConfig::small()),
        Command::TrainCustom {
            corpus,
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            mlp_dim,
            max_seq_len,
        } => pretrain(
            corpus,
            MetaIConfig {
                vocab_size,
                hidden_dim,
                num_layers,
                num_heads,
                num_kv_heads,
                mlp_dim,
                max_seq_len,
                ..MetaIConfig::tiny()
            },
        ),
        Command::Lab {
            kind,
            output_dir,
            schedule,
        } => metai::labs::run(kind, &output_dir, &schedule),
        Command::PlayAgent {
            model_dir,
            case,
            imitation,
        } => {
            println!("{}", metai::labs::agent::play(&model_dir, case, imitation)?);
            Ok(())
        }
        Command::TrainAgent {
            model_dir,
            output_dir,
            schedule,
        } => metai::labs::agent::run(&model_dir, &output_dir, &schedule),
        Command::TrainDecision {
            post: p,
            calibration_path,
            pooling,
            test_path,
        } => metai::decision::train_with_pooling(
            &p.model_dir,
            p.data_path.as_ref(),
            &calibration_path,
            &test_path,
            &p.output_dir,
            &p.schedule,
            pooling,
        ),
        Command::Decide { model_dir, text } => {
            println!("{}", metai::decision::predict(&model_dir, &text)?);
            Ok(())
        }
        Command::TrainThinking {
            post: p,
            budget,
            cost,
        } => train::thinking::train(
            &p.data_path,
            &p.model_dir,
            &p.output_dir,
            &p.schedule,
            budget,
            cost,
        ),
        Command::Think {
            model_dir,
            prompt,
            max_len,
            threshold,
            budget,
        } => {
            println!(
                "{}",
                train::thinking::generate(&model_dir, &prompt, max_len, threshold, budget)?
            );
            Ok(())
        }
        Command::TrainMultimodal {
            post: p,
            audio,
            train_readout,
            continuation_targets,
        } => metai::labs::multimodal::train_with_options(
            &p.model_dir,
            p.data_path.as_ref(),
            &p.output_dir,
            audio,
            train_readout,
            continuation_targets,
            &p.schedule,
        ),
        Command::GenerateMultimodal {
            model_dir,
            data_path,
            max_len,
        } => metai::labs::multimodal::generate(&model_dir, &data_path, max_len),
        Command::TrainSft(p) => {
            train::sft::run(&p.data_path, &p.model_dir, &p.output_dir, &p.schedule)
        }
        Command::TrainDpo { post: p, beta } => {
            train::dpo::run(&p.data_path, &p.model_dir, &p.output_dir, &p.schedule, beta)
        }
        Command::TrainLora {
            post: p,
            rank,
            alpha,
        } => train::adaptation::lora(
            &p.data_path,
            &p.model_dir,
            &p.output_dir,
            &p.schedule,
            rank,
            alpha,
        ),
        Command::Distill {
            post: p,
            temperature,
        } => train::adaptation::distill(
            &p.data_path,
            &p.model_dir,
            &p.output_dir,
            &p.schedule,
            temperature,
        ),
        Command::TrainGrpo { post: p, rollout } => train::grpo_train_step::run(
            &p.data_path,
            &p.model_dir,
            &p.output_dir,
            &p.schedule,
            train::grpo::Algorithm::Grpo,
            &rollout,
        ),
        Command::TrainGspo { post: p, rollout } => train::grpo_train_step::run(
            &p.data_path,
            &p.model_dir,
            &p.output_dir,
            &p.schedule,
            train::grpo::Algorithm::Gspo,
            &rollout,
        ),
        Command::TrainPpo { post: p, rollout } => train::grpo_train_step::run(
            &p.data_path,
            &p.model_dir,
            &p.output_dir,
            &p.schedule,
            train::grpo::Algorithm::Ppo,
            &rollout,
        ),
        Command::TrainCispo { post: p, rollout } => train::grpo_train_step::run(
            &p.data_path,
            &p.model_dir,
            &p.output_dir,
            &p.schedule,
            train::grpo::Algorithm::Cispo,
            &rollout,
        ),
        Command::Generate {
            model_dir,
            prompt,
            chat,
            max_len,
            temperature,
            top_k,
            top_p,
        } => {
            let ck = Checkpoint::load(&model_dir, &get_device())?;
            let prompt = if chat {
                metai::data::chat_prompt(&prompt, "")
            } else {
                prompt
            };
            ensure!(
                temperature.is_finite() && temperature >= 0. && top_p > 0. && top_p <= 1.,
                "invalid sampling parameters"
            );
            let n = ck.tokenizer.encode(&prompt).len();
            ensure!(
                n > 0 && n.saturating_add(max_len) <= ck.config.max_seq_len,
                "prompt and generation budget exceed context or prompt is empty"
            );
            println!(
                "{}",
                Generator::new(ck.model, ck.tokenizer).generate(
                    &prompt,
                    max_len,
                    temperature,
                    top_k,
                    top_p
                )
            );
            Ok(())
        }
        Command::EvalChat {
            model_dir,
            questions,
            max_len,
        } => {
            use std::io::{BufRead, BufReader};
            #[derive(serde::Deserialize)]
            struct Question {
                id: String,
                instruction: String,
                #[serde(default)]
                input: String,
            }
            ensure!(max_len > 0, "generation budget must be positive");
            let ck = Checkpoint::load(&model_dir, &get_device())?;
            let tokenizer = ck.tokenizer.clone();
            let generator = Generator::new(ck.model, ck.tokenizer);
            for line in BufReader::new(std::fs::File::open(questions)?).lines() {
                let q: Question = serde_json::from_str(&line?)?;
                let ids = tokenizer.encode(&metai::data::chat_prompt(&q.instruction, &q.input));
                ensure!(
                    !ids.is_empty() && ids.len().saturating_add(max_len) <= ck.config.max_seq_len,
                    "question {} and generation budget exceed context",
                    q.id
                );
                let tokens = generator.completion_ids(&ids, max_len, 0., 0, 1.);
                let last = tokens.last().copied();
                let stop = if last == tokenizer.eos_id() {
                    "eos"
                } else if last == tokenizer.pad_id() {
                    "pad"
                } else {
                    "budget"
                };
                println!(
                    "{}",
                    serde_json::json!({"id": q.id, "answer": tokenizer.decode(&tokens),
                    "token_ids": tokens, "stop": stop})
                );
            }
            Ok(())
        }
        Command::GenerateSpec {
            model_dir,
            draft_dir,
            prompt,
            max_len,
            lookahead,
        } => {
            let ck = Checkpoint::load(&model_dir, &get_device())?;
            let draft = Checkpoint::load(&draft_dir, &get_device())?;
            ensure!(
                lookahead > 0
                    && !prompt.is_empty()
                    && ck.tokenizer.encode(&prompt).len().saturating_add(max_len)
                        <= ck.config.max_seq_len.min(draft.config.max_seq_len),
                "invalid speculation budget"
            );
            ensure!(
                ck.tokenizer.equivalent(&draft.tokenizer)?,
                "draft and target tokenizers differ"
            );
            println!(
                "{}",
                SpeculativeGenerator::new(draft.model, ck.model, ck.tokenizer)
                    .generate(&prompt, max_len, lookahead, 0.)
            );
            Ok(())
        }
        Command::Quantize {
            input_path,
            output_path,
        } => {
            let device = get_device();
            let mut ck = Checkpoint::load(&input_path, &device)?;
            ck.model = ck.model.quantize_int4(&device);
            ck.save(&output_path)
        }
    }
}
