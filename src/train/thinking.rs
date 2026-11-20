//! A tied latent refinement layer with a learned per-token halting probability.
use super::*;
use crate::data::sft::{InstructionDataset, SFTBatcher};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    nn::{Linear, LinearConfig},
    tensor::activation::sigmoid,
};

#[derive(Module, Debug)]
struct Thinker {
    refine: Linear,
    halt: Linear,
}
impl Thinker {
    fn new(width: usize, d: &Device) -> Self {
        Self {
            refine: LinearConfig::new(width, width).init(d),
            halt: LinearConfig::new(width, 1).init(d),
        }
    }
    fn advance(&self, h: Tensor<3>) -> Tensor<3> {
        h.clone() + self.refine.forward(h).tanh()
    }
}

pub fn train(
    data: &str,
    base: &Path,
    out: &Path,
    s: &Schedule,
    budget: usize,
    cost: f64,
) -> Result<()> {
    ensure!(
        (1..=32).contains(&budget) && cost.is_finite() && cost >= 0.,
        "invalid thinking budget or cost"
    );
    let device = crate::backend::get_device().autodiff();
    device.seed(s.seed);
    let ck = Checkpoint::load(base, &device)?;
    let frozen = ck.model.clone().no_grad();
    let ds = InstructionDataset::from_file(data, &ck.tokenizer, ck.config.max_seq_len)?;
    let batcher = SFTBatcher::new(device.clone(), ck.model.pad_id);
    let thinker = s.fit(
        Thinker::new(ck.config.hidden_dim, &device),
        ds.len(),
        |m, ids| {
            let b = batcher.batch(rows(&ds, ids), &device);
            let dims = b.inputs.dims();
            let mut h = frozen.hidden(b.inputs, None, None).detach();
            let mut alive = Tensor::<2>::ones(dims, &device);
            let mut expected = Tensor::zeros(dims, &device);
            for step in 1..=budget {
                h = m.advance(h);
                let stop = if step == budget {
                    alive.clone()
                } else {
                    alive.clone() * sigmoid(m.halt.forward(h.clone())).reshape(dims)
                };
                let nll = -token_logps(frozen.output.forward(h.clone()), b.targets.clone());
                expected = expected + stop.clone() * (nll + cost * step as f64);
                alive = alive - stop;
            }
            let mask = b.mask.float();
            Ok((expected * mask.clone()).sum() / mask.sum())
        },
    )?;
    ck.save_with(out, |dir| {
        thinker.save_file(dir.join("thinker.bpk"))?;
        fs::write(dir.join("budget.json"), serde_json::to_vec(&budget)?)?;
        Ok(())
    })
}

pub fn generate(
    dir: &Path,
    prompt: &str,
    max_new: usize,
    threshold: f64,
    budget: Option<usize>,
) -> Result<serde_json::Value> {
    ensure!(
        threshold.is_finite() && threshold > 0. && threshold <= 1.,
        "threshold must be in (0,1]"
    );
    let device = crate::backend::get_device();
    let ck = Checkpoint::load(dir, &device)?;
    let trained: usize = serde_json::from_reader(fs::File::open(dir.join("budget.json"))?)?;
    let budget = budget.unwrap_or(trained);
    ensure!(
        budget > 0 && budget <= trained,
        "budget exceeds trained depth"
    );
    let thinker =
        Thinker::new(ck.config.hidden_dim, &device).try_load_file(dir.join("thinker.bpk"))?;
    ensure!(!prompt.trim().is_empty(), "empty question");
    let mut ids = ck.tokenizer.encode(&crate::data::chat_prompt(prompt, ""));
    let answer_start = ids.len();
    let mut steps = Vec::new();
    ensure!(
        !ids.is_empty() && ids.len().saturating_add(max_new) <= ck.config.max_seq_len,
        "invalid context length"
    );
    for _ in 0..max_new {
        let n = ids.len();
        let input: Vec<_> = ids.iter().map(|&v| v as i32).collect();
        let mut h = ck
            .model
            .hidden(
                Tensor::<1, Int>::from_ints(input.as_slice(), &device).reshape([1, n]),
                None,
                None,
            )
            .slice([0..1, n - 1..n]);
        let mut survival = 1.;
        for step in 1..=budget {
            h = thinker.advance(h);
            survival *= 1. - sigmoid(thinker.halt.forward(h.clone())).into_scalar::<f64>();
            if 1. - survival >= threshold || step == budget {
                steps.push(step);
                break;
            }
        }
        let token = ck.model.output.forward(h).argmax(2).into_scalar::<u32>();
        ids.push(token);
        if Some(token) == ck.tokenizer.eos_id() {
            break;
        }
    }
    Ok(
        serde_json::json!({"text":ck.tokenizer.decode(&ids[answer_start..]),"latent_steps_per_token":steps,"budget":budget}),
    )
}
