//! Score the same target translation under the original and a rotated source sentence.
use anyhow::{ensure, Context, Result};
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{Int, Tensor};
use clap::Parser;
use metai::data::sft::{encode_instruction, InstructionItem, SFTBatcher};
use metai::train::{token_logps, Checkpoint};
use metai::Generator;
use serde::Deserialize;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    contrast: PathBuf,
}

#[derive(Deserialize)]
struct Contrast {
    id: String,
    instruction: String,
    input: String,
    swapped_input: String,
    output: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = metai::backend::get_device();
    let ck = Checkpoint::load(&args.model_dir, &device)?;
    let batcher = SFTBatcher::new(device.clone(), ck.model.pad_id);
    let generator = Generator::new(ck.model.clone(), ck.tokenizer.clone());
    for (i, line) in BufReader::new(File::open(args.contrast)?)
        .lines()
        .enumerate()
    {
        let c: Contrast = serde_json::from_str(&line?)?;
        let mut records = Vec::new();
        let target = ck.tokenizer.response(&c.output);
        ensure!(target.len() > 1, "empty translation {}", c.id);
        for source in [c.input, c.swapped_input] {
            let row = encode_instruction(
                &InstructionItem {
                    instruction: c.instruction.clone(),
                    input: Some(source),
                    output: c.output.clone(),
                },
                &ck.tokenizer,
                ck.config.max_seq_len,
            )
            .context("empty supervised sequence")?;
            ensure!(row.0.ends_with(&target), "truncated target {}", c.id);
            records.push(row);
        }
        let prefix = records[0].0.len() - target.len();
        let swapped_prefix = records[1].0.len() - target.len();
        ensure!(
            records[0].0[..prefix] != records[1].0[..swapped_prefix],
            "identical encoded prompts {}",
            c.id
        );
        // The first two development items cover both directions. Recompute full prefixes
        // to check whether the measured greedy output is an artifact of the KV cache.
        let cache_check = if i < 2 {
            let prompt = &records[0].0[..prefix];
            let cached = generator.completion_ids(prompt, 32, 0., 0, 1.);
            let mut ids: Vec<i32> = prompt.iter().map(|&id| id as i32).collect();
            let mut uncached = Vec::new();
            for _ in 0..32 {
                let n = ids.len();
                let x = Tensor::<1, Int>::from_ints(ids.as_slice(), &device).reshape([1, n]);
                let next = ck
                    .model
                    .forward(x, None, None)
                    .slice([0..1, n - 1..n])
                    .argmax(2)
                    .into_scalar::<u32>();
                uncached.push(next);
                if Some(next) == ck.tokenizer.eos_id() || Some(next) == ck.tokenizer.pad_id() {
                    break;
                }
                ids.push(next as i32);
            }
            Some(
                serde_json::json!({"cached": cached, "uncached": uncached, "equal": cached==uncached}),
            )
        } else {
            None
        };
        let batch = batcher.batch(records, &device);
        let logps = token_logps(ck.model.forward(batch.inputs, None, None), batch.targets);
        let losses = (-(logps * batch.mask.float()).sum_dim(1) / (target.len() as f32))
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        ensure!(
            losses.len() == 2 && losses.iter().all(|x| x.is_finite()),
            "invalid losses"
        );
        println!(
            "{}",
            serde_json::json!({"id":c.id, "targets":target.len(),"cache_check":cache_check,"matched_nll":losses[0],"swapped_nll":losses[1]})
        );
    }
    Ok(())
}
