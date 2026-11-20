//! Output-projection LoRA and temperature-scaled online logit distillation.
use super::*;
use crate::backend::get_device;
use crate::data::{
    data::{TextBatcher, TextDataset},
    sft::{InstructionDataset, SFTBatcher},
};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    nn::{Initializer, Linear, LinearConfig},
};

#[derive(Module, Debug)]
struct Adapter {
    down: Linear,
    up: Linear,
}

pub fn lora(
    data: &str,
    dir: &Path,
    output: &Path,
    schedule: &Schedule,
    rank: usize,
    alpha: f64,
) -> Result<()> {
    let device = get_device().autodiff();
    device.seed(schedule.seed);
    let mut ck = Checkpoint::load(dir, &device)?;
    ensure!(
        rank > 0 && rank <= ck.config.hidden_dim && alpha.is_finite() && alpha > 0.,
        "invalid adapter rank or alpha"
    );
    let adapter = Adapter {
        down: LinearConfig::new(ck.config.hidden_dim, rank)
            .with_bias(false)
            .init(&device),
        up: LinearConfig::new(rank, ck.config.vocab_size)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(&device),
    };
    let ds = InstructionDataset::from_file(data, &ck.tokenizer, ck.config.max_seq_len)?;
    let batcher = SFTBatcher::new(device.clone(), ck.model.pad_id);
    let scale = alpha / rank as f64;
    let base = ck.model.clone().no_grad();
    let adapter = schedule.fit(adapter, ds.len(), |a, ids| {
        let b = batcher.batch(rows(&ds, ids), &device);
        let h = base.hidden(b.inputs, None, None).detach();
        let logits = base.output.forward(h.clone()) + a.up.forward(a.down.forward(h)) * scale;
        Ok(masked_nll(logits, b.targets, b.mask))
    })?;
    // Burn's live Linear weight is [in, out]. Keep the load/save layout mappers.
    let delta = adapter.down.weight.val().matmul(adapter.up.weight.val()) * scale;
    ck.model.output.weight = ck.model.output.weight.map(|w| w + delta);
    println!(
        "adapter_parameters={} rank={rank} output=merged",
        rank * (ck.config.hidden_dim + ck.config.vocab_size)
    );
    ck.save_with(output, |dir| {
        adapter.save_file(dir.join("adapter.bpk"))?;
        fs::write(
            dir.join("adapter.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"target":"output","rank":rank,"alpha":alpha,"merged":true}),
            )?,
        )?;
        Ok(())
    })
}

pub fn distill(data: &str, dir: &Path, output: &Path, schedule: &Schedule, t: f64) -> Result<()> {
    ensure!(t.is_finite() && t > 0., "temperature must be positive");
    let device = get_device().autodiff();
    device.seed(schedule.seed);
    let teacher = Checkpoint::load(dir, &device)?;
    let mut config = MetaIConfig::tiny();
    config.hidden_dim = (teacher.config.hidden_dim / 2).max(4) / 4 * 4;
    config.vocab_size = teacher.config.vocab_size;
    config.max_seq_len = teacher.config.max_seq_len;
    config.mlp_dim = config.hidden_dim * 2;
    config.num_experts = 1;
    config.active_experts = 1;
    let pad = teacher.model.pad_id;
    let model = MetaIModel::new(&config, pad, &device);
    let ds = TextDataset::from_file(data, &teacher.tokenizer, config.max_seq_len)?;
    let batcher = TextBatcher::new(device.clone(), pad);
    let frozen = teacher.model.no_grad();
    let student = schedule.fit(model, ds.len(), |m, ids| {
        let b = batcher.batch(rows(&ds, ids), &device);
        let q = log_softmax(frozen.forward(b.inputs.clone(), None, None) / t, 2).detach();
        let p = log_softmax(m.forward(b.inputs, None, None) / t, 2);
        let mask = b.targets.not_equal_elem(pad as i32).float();
        let kl = (q.clone().exp() * (q - p)).sum_dim(2).squeeze_dim::<2>(2);
        Ok((kl * mask.clone()).sum() / mask.sum() * (t * t))
    })?;
    Checkpoint {
        config,
        model: student,
        tokenizer: teacher.tokenizer,
    }
    .save(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trained_low_rank_update_merges_into_the_same_projection() {
        let d = get_device();
        let mut base = LinearConfig::new(8, 12).with_bias(false).init(&d);
        let adapter = Adapter {
            down: LinearConfig::new(8, 2).with_bias(false).init(&d),
            up: LinearConfig::new(2, 12).with_bias(false).init(&d),
        };
        let x = Tensor::<3>::random([2, 4, 8], burn::tensor::Distribution::Default, &d);
        let expected =
            base.forward(x.clone()) + adapter.up.forward(adapter.down.forward(x.clone())) * 3.;
        let delta = adapter.down.weight.val().matmul(adapter.up.weight.val()) * 3.;
        base.weight = base.weight.map(|w| w + delta);
        assert!(
            (expected - base.forward(x))
                .abs()
                .max()
                .into_scalar::<f32>()
                < 1e-5
        );
    }
}
