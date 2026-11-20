use super::*;
use crate::data::sft::{InstructionDataset, SFTBatcher};
use burn::data::{dataloader::batcher::Batcher, dataset::Dataset};

pub fn run(data: &str, dir: &Path, output: &Path, schedule: &Schedule) -> Result<()> {
    let device = crate::backend::get_device().autodiff();
    let mut ck = Checkpoint::load(dir, &device)?;
    let ds = InstructionDataset::from_file(data, &ck.tokenizer, ck.config.max_seq_len)?;
    let batcher = SFTBatcher::new(device.clone(), ck.model.pad_id);
    ck.model = schedule.fit(ck.model, ds.len(), |m, ids| {
        let b = batcher.batch(rows(&ds, ids), &device);
        Ok(masked_nll(
            m.forward(b.inputs, None, None),
            b.targets,
            b.mask,
        ))
    })?;
    let report = serde_json::json!({
        "runtime": format!("{:?}", crate::backend::runtime()), "parameters": ck.model.num_params(),
        "examples": ds.len(), "supervised_targets_per_epoch": ds.supervised_targets(),
        "schedule": schedule, "optimizer_steps": ds.len().div_ceil(schedule.batch_size) * schedule.num_epochs
    });
    ck.save_with(output, |dir| {
        fs::write(
            dir.join("training-report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        Ok(())
    })
}
