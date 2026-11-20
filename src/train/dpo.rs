use super::*;
use crate::data::dpo::{DPOBatch, DPOBatcher, PreferenceDataset};
use burn::data::{dataloader::batcher::Batcher, dataset::Dataset};
use burn::tensor::activation::log_sigmoid;

fn margin(model: &MetaIModel, batch: DPOBatch) -> Tensor<1> {
    let score = |x, y, mask: Tensor<2, Int>| {
        (token_logps(model.forward(x, None, None), y) * mask.float())
            .sum_dim(1)
            .squeeze_dim(1)
    };
    score(batch.chosen_inputs, batch.chosen_targets, batch.chosen_mask)
        - score(
            batch.rejected_inputs,
            batch.rejected_targets,
            batch.rejected_mask,
        )
}

pub fn run(data: &str, dir: &Path, output: &Path, schedule: &Schedule, beta: f64) -> Result<()> {
    ensure!(beta.is_finite() && beta > 0., "beta must be positive");
    let device = crate::backend::get_device().autodiff();
    let mut ck = Checkpoint::load(dir, &device)?;
    let ds = PreferenceDataset::from_file(data, &ck.tokenizer, ck.config.max_seq_len)?;
    let batcher = DPOBatcher::new(device.clone(), ck.model.pad_id);
    let reference = ck.model.clone().no_grad();
    ck.model = schedule.fit(ck.model, ds.len(), |policy, ids| {
        let b = batcher.batch(rows(&ds, ids), &device);
        let baseline = margin(&reference, b.clone()).detach();
        Ok(-log_sigmoid((margin(policy, b) - baseline) * beta).mean())
    })?;
    ck.save(output)
}
