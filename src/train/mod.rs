//! Training and checkpoint exchange. All stages exchange the same self-contained model directory.
use anyhow::{ensure, Context, Result};
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::{activation::log_softmax, Device, Int, Tensor};
use std::{fs, path::Path};

use crate::{MetaIConfig, MetaIModel, MetaITokenizer};

pub mod adaptation;
pub mod distributed;
pub mod dpo;
pub mod grpo;
pub mod grpo_rollout;
pub mod grpo_train_step;
pub mod reward;
pub mod sft;
pub mod thinking;
#[allow(clippy::module_inception)] // Preserve the existing public module path.
pub mod train;

pub struct Checkpoint {
    pub config: MetaIConfig,
    pub model: MetaIModel,
    pub tokenizer: MetaITokenizer,
}

impl Checkpoint {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let config: MetaIConfig = serde_json::from_reader(
            fs::File::open(dir.join("config.json")).with_context(|| {
                format!(
                    "Missing config.json in {} (expected a Burn 0.22 checkpoint)",
                    dir.display()
                )
            })?,
        )?;
        config.validate()?;
        let tokenizer = MetaITokenizer::new(dir.join("tokenizer.json"))?;
        ensure!(
            tokenizer.vocab_size() <= config.vocab_size,
            "tokenizer exceeds model vocabulary"
        );
        let model = MetaIModel::new(
            &config,
            tokenizer.pad_id().context("missing pad token")?,
            device,
        )
        .try_load_file(dir.join("model.bpk"))?;
        Ok(Self {
            model,
            config,
            tokenizer,
        })
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        self.save_with(dir, |_| Ok(()))
    }

    pub fn save_with(&self, dir: &Path, extra: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
        // Publish the directory only after all three files exist.
        let parent = dir
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let staging = tempfile::Builder::new()
            .prefix(".metai-")
            .tempdir_in(parent)?;
        self.model
            .clone()
            .save_file(staging.path().join("model.bpk"))?;
        fs::write(
            staging.path().join("config.json"),
            serde_json::to_vec_pretty(&self.config)?,
        )?;
        self.tokenizer.save(staging.path().join("tokenizer.json"))?;
        extra(staging.path())?;
        ensure!(!dir.exists(), "output already exists: {}", dir.display());
        fs::rename(staging.path(), dir)?;
        Ok(())
    }
}

pub fn token_logps(logits: Tensor<3>, targets: Tensor<2, Int>) -> Tensor<2> {
    let [b, s] = targets.dims();
    log_softmax(logits, 2)
        .gather(2, targets.reshape([b, s, 1]))
        .reshape([b, s])
}

pub fn masked_nll(logits: Tensor<3>, targets: Tensor<2, Int>, mask: Tensor<2, Int>) -> Tensor<1> {
    let weights = mask.float();
    -(token_logps(logits, targets) * weights.clone()).sum() / weights.sum()
}

#[derive(Clone, Debug, clap::Args, serde::Serialize, serde::Deserialize)]
pub struct Schedule {
    #[arg(long, default_value_t = 1)]
    pub num_epochs: usize,
    #[arg(long, default_value_t = 4)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 0.001)]
    pub learning_rate: f64,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

impl Schedule {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.num_epochs > 0
                && self.batch_size > 0
                && self.learning_rate.is_finite()
                && self.learning_rate > 0.0,
            "epochs, batch size and learning rate must be positive"
        );
        Ok(())
    }

    /// A batch closure owns the task-specific data and loss. Only the policy enters AdamW.
    pub fn fit<M: Module>(
        &self,
        mut model: M,
        n: usize,
        mut loss_for: impl FnMut(&M, &[usize]) -> Result<Tensor<1>>,
    ) -> Result<M> {
        self.validate()?;
        let mut optimizer = AdamWConfig::new().init();
        for epoch in 1..=self.num_epochs {
            model = self.epoch(model, n, epoch, &mut optimizer, &mut loss_for)?;
        }
        Ok(model)
    }

    fn epoch<M: Module>(
        &self,
        mut model: M,
        n: usize,
        epoch: usize,
        optimizer: &mut burn::optim::ModuleOptimizer,
        mut loss_for: impl FnMut(&M, &[usize]) -> Result<Tensor<1>>,
    ) -> Result<M> {
        use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
        ensure!(
            n > 0,
            "no trainable examples after tokenization and truncation"
        );
        let mut order: Vec<_> = (0..n).collect();
        order.shuffle(&mut StdRng::seed_from_u64(
            self.seed.wrapping_add((epoch - 1) as u64),
        ));
        let mut total = 0.;
        let log_interval = std::env::var("METAI_LOG_EVERY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0);
        for (index, batch) in order.chunks(self.batch_size).enumerate() {
            let loss = loss_for(&model, batch)?;
            let value = loss.clone().into_scalar::<f32>();
            ensure!(value.is_finite(), "nonfinite loss in epoch {epoch}");
            total += value as f64 * batch.len() as f64;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(self.learning_rate, model, grads);
            if log_interval.is_some_and(|every| (index + 1).is_multiple_of(every)) {
                eprintln!(
                    "epoch={epoch} step={}/{} batch_nll={value:.6}",
                    index + 1,
                    n.div_ceil(self.batch_size)
                );
            }
        }
        println!(
            "epoch={epoch} examples={n} mean_batch_loss={:.6}",
            total / n as f64
        );
        Ok(model)
    }
}

pub fn rows<I: Clone + Send + Sync>(
    data: &impl burn::data::dataset::Dataset<I>,
    indices: &[usize],
) -> Vec<I> {
    indices
        .iter()
        .map(|&i| data.get(i).expect("indices drawn from dataset length"))
        .collect()
}
