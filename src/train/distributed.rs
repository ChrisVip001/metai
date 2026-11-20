//! Single-process synchronous data parallelism. Rank zero owns AdamW and checkpoint state.
use super::{token_logps, Schedule};
use crate::{
    backend::device_for_rank,
    data::data::{LazyTextDataset, TextBatcher},
    MetaIModel,
};
use anyhow::{ensure, Context, Result};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    module::Module,
    optim::{GradientsAccumulator, GradientsParams, ModuleOptimizer},
    tensor::Device,
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Topology {
    pub world_size: usize,
    pub accumulation_steps: usize,
}

impl Topology {
    pub fn global_batch(self, micro: usize) -> Result<usize> {
        ensure!(
            self.world_size > 0 && micro > 0 && self.accumulation_steps > 0,
            "world size, micro-batch and accumulation steps must be positive"
        );
        self.world_size
            .checked_mul(micro)
            .and_then(|n| n.checked_mul(self.accumulation_steps))
            .context("global batch size overflow")
    }
}

pub(super) fn epoch(
    mut model: MetaIModel,
    data: &LazyTextDataset,
    schedule: &Schedule,
    epoch: usize,
    optimizer: &mut ModuleOptimizer,
    topology: Topology,
) -> Result<MetaIModel> {
    let micro = schedule.batch_size;
    let global = topology.global_batch(micro)?;
    let mut order: Vec<_> = (0..data.len()).collect();
    ensure!(!order.is_empty(), "no training blocks");
    order.shuffle(&mut StdRng::seed_from_u64(
        schedule.seed.wrapping_add((epoch - 1) as u64),
    ));
    let primary = device_for_rank(0, topology.world_size).autodiff();
    let mut epoch_loss = 0.;
    let mut epoch_tokens = 0;
    let mut steps = 0;
    for batch in order.chunks(global) {
        let tokens: usize = batch.iter().map(|&i| data.targets(i)).sum();
        let mut reduced = GradientsAccumulator::new();
        // Every sample is scheduled once; an incomplete final wave starts only the ranks with data.
        for wave in batch.chunks(topology.world_size * micro) {
            let results = std::thread::scope(|scope| -> Result<Vec<_>> {
                let mut jobs = Vec::new();
                for (rank, indices) in wave.chunks(micro).enumerate() {
                    let device = device_for_rank(rank, topology.world_size).autodiff();
                    let replica = model.clone().fork(&device);
                    jobs.push(
                        scope.spawn(move || backward(replica, data, indices, tokens, &device)),
                    );
                }
                jobs.into_iter()
                    .map(|job| {
                        job.join()
                            .map_err(|_| anyhow::anyhow!("training rank panicked"))?
                    })
                    .collect()
            })?;
            for (nll, grads) in results {
                epoch_loss += nll;
                reduced.accumulate(&model, grads.to_device(&primary.clone().inner(), &model));
            }
        }
        model = optimizer.step(schedule.learning_rate, model, reduced.grads());
        steps += 1;
        epoch_tokens += tokens;
    }
    eprintln!("epoch={epoch} ranks={} global_batch={global} optimizer_steps={steps} train_tokens={epoch_tokens} train_nll={:.6}",
             topology.world_size, epoch_loss / epoch_tokens as f64);
    Ok(model)
}

fn backward(
    model: MetaIModel,
    data: &LazyTextDataset,
    indices: &[usize],
    global_tokens: usize,
    device: &Device,
) -> Result<(f64, GradientsParams)> {
    let batch =
        TextBatcher::new(device.clone(), model.pad_id).batch(data.batch_rows(indices)?, device);
    let mask = batch
        .targets
        .clone()
        .not_equal_elem(model.pad_id as i32)
        .float();
    let loss = -(token_logps(model.forward(batch.inputs, None, None), batch.targets) * mask).sum();
    let nll = loss.clone().into_scalar::<f32>() as f64;
    ensure!(nll.is_finite(), "nonfinite loss on rank");
    let grads = GradientsParams::from_grads((loss / global_tokens as f32).backward(), &model);
    Ok((nll, grads))
}

/// Published V4 sizes are rounded. This is a capacity calculation, not an executable V4 configuration.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum CapacityModel {
    #[value(name = "125m")]
    Prototype125m,
    V4Flash,
    V4Pro,
}

#[derive(Debug, clap::Args)]
pub struct CapacityPlan {
    #[arg(long, value_enum)]
    pub model: CapacityModel,
    /// Total physical accelerators across all replicas.
    #[arg(long)]
    pub world_size: u64,
    /// Accelerators assigned to one full model replica. Assumes ideal uniform weight placement.
    #[arg(long, default_value_t = 1)]
    pub weight_shards: u64,
    /// Packed weight bits; excludes block metadata and higher-precision layers.
    #[arg(long, default_value_t = 16)]
    pub weight_bits: u8,
    /// Samples processed by one complete model replica per micro-step.
    #[arg(long, default_value_t = 1)]
    pub micro_batch: u64,
    #[arg(long, default_value_t = 1)]
    pub accumulation_steps: u64,
    /// Optional packed sequence length for the tokens-per-update budget.
    #[arg(long)]
    pub sequence_length: Option<u64>,
}

impl CapacityPlan {
    pub fn report(&self) -> Result<serde_json::Value> {
        let (name, params, active, rounded): (&str, u64, Option<u64>, bool) = match self.model {
            CapacityModel::Prototype125m => ("MetaI-125M", 125_276_928, None, false),
            CapacityModel::V4Flash => (
                "DeepSeek-V4-Flash",
                284_000_000_000,
                Some(13_000_000_000),
                true,
            ),
            CapacityModel::V4Pro => (
                "DeepSeek-V4-Pro",
                1_600_000_000_000,
                Some(49_000_000_000),
                true,
            ),
        };
        ensure!(
            self.world_size > 0
                && self.weight_shards > 0
                && self.world_size.is_multiple_of(self.weight_shards),
            "weight shards must divide the accelerator count into full replicas"
        );
        ensure!(
            matches!(self.weight_bits, 4 | 8 | 16 | 32),
            "supported packed weight widths: 4, 8, 16, 32 bits"
        );
        ensure!(
            self.micro_batch > 0 && self.accumulation_steps > 0 && self.sequence_length != Some(0),
            "micro-batch, accumulation and sequence length must be positive"
        );
        let replicas = self.world_size / self.weight_shards;
        let batch = replicas
            .checked_mul(self.micro_batch)
            .and_then(|n| n.checked_mul(self.accumulation_steps))
            .context("global batch overflow")?;
        let token_slots = self
            .sequence_length
            .map(|s| batch.checked_mul(s).context("token budget overflow"))
            .transpose()?;
        let bytes = (params * u64::from(self.weight_bits)).div_ceil(8);
        Ok(serde_json::json!({
            "model": name, "parameters": params, "active_parameters_reference": active,
            "rounded_parameter_count": rounded, "world_size": self.world_size, "weight_shards_per_replica":self.weight_shards,
            "replicas":replicas, "weight_bits":self.weight_bits,
            "packed_weight_bytes_per_replica":bytes, "ideal_even_shard_bytes":bytes.div_ceil(self.weight_shards),
            "global_samples_per_update":batch, "sequence_length":self.sequence_length, "token_slots_per_full_update":token_slots,
            "excluded":["replicated layers","quantization metadata","optimizer states","gradients","activations","KV cache","communication buffers","allocator overhead"]
        }))
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::*;
    #[test]
    fn capacity_uses_total_weights_and_counts_only_independent_replicas_in_batch() {
        let mut plan = CapacityPlan {
            model: CapacityModel::V4Pro,
            world_size: 128,
            weight_shards: 64,
            weight_bits: 4,
            micro_batch: 1,
            accumulation_steps: 256,
            sequence_length: Some(1_000_000),
        };
        let r = plan.report().unwrap();
        assert_eq!(r["packed_weight_bytes_per_replica"], 800_000_000_000_u64);
        assert_eq!(r["ideal_even_shard_bytes"], 12_500_000_000_u64);
        assert_eq!(r["token_slots_per_full_update"], 512_000_000);
        plan.model = CapacityModel::Prototype125m;
        let config = crate::MetaIConfig::prototype_125m();
        let measured = MetaIModel::new(&config, 0, &crate::backend::get_device()).num_params();
        assert_eq!(plan.report().unwrap()["parameters"], measured as u64);
        plan.weight_shards = 72;
        assert!(plan.report().is_err());
        plan.weight_shards = 0;
        assert!(plan.report().is_err());
    }
}
