use super::distributed::Topology;
use super::*;
use crate::backend::get_device;
use crate::data::data::{LazyTextDataset, TextBatcher};
use burn::data::{dataloader::batcher::Batcher, dataset::Dataset};

#[derive(serde::Serialize, serde::Deserialize)]
struct Progress {
    epoch: usize,
    schedule: Schedule,
    corpus_hashes: [String; 2],
    #[serde(default)]
    extra_train_hashes: Vec<String>,
    #[serde(default)]
    parallel: Option<Topology>,
}
fn digest(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    std::io::copy(&mut fs::File::open(path)?, &mut h)?;
    Ok(format!("{:x}", h.finalize()))
}

#[derive(Clone, Copy)]
pub struct Sources<'a> {
    pub train: &'a [&'a Path],
    pub valid: Option<&'a Path>,
}

impl Sources<'_> {
    fn fingerprints(self) -> Result<([String; 2], Vec<String>)> {
        let (first, rest) = self.train.split_first().context("no training corpus")?;
        let hashes = [
            digest(first)?,
            self.valid.map(digest).transpose()?.unwrap_or_default(),
        ];
        let extra = rest.iter().map(|p| digest(p)).collect::<Result<Vec<_>>>()?;
        Ok((hashes, extra))
    }
}

/// Tokenizer training and the block reader must see the same plain-text stream.
fn text_source(path: &Path, scratch: &Path, slot: usize) -> Result<std::path::PathBuf> {
    use std::io::{BufRead, BufReader, BufWriter, Write};
    if path.extension().is_none_or(|ext| ext != "jsonl") {
        return Ok(path.to_owned());
    }
    #[derive(serde::Deserialize)]
    struct Document {
        text: String,
    }
    let dest = scratch.join(format!("source-{slot}.txt"));
    let mut output = BufWriter::new(fs::File::create(&dest)?);
    for (line_num, line) in BufReader::new(fs::File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let doc: Document = serde_json::from_str(&line).with_context(|| {
            format!(
                "{}:{} expected a JSON object with a text string",
                path.display(),
                line_num + 1
            )
        })?;
        writeln!(output, "{}", doc.text)?;
    }
    output.flush()?;
    Ok(dest)
}

struct TextSources {
    train: Vec<std::path::PathBuf>,
    valid: Option<std::path::PathBuf>,
}
impl TextSources {
    fn prepare(sources: Sources, scratch: &Path) -> Result<Self> {
        Ok(Self {
            train: sources
                .train
                .iter()
                .enumerate()
                .map(|(i, p)| text_source(p, scratch, i))
                .collect::<Result<_>>()?,
            valid: sources
                .valid
                .map(|p| text_source(p, scratch, sources.train.len()))
                .transpose()?,
        })
    }
}
pub fn resume(
    base: &Path,
    train: &str,
    valid: &str,
    out: &Path,
    additional: usize,
    snapshots: Option<&Path>,
) -> Result<()> {
    resume_sources(
        base,
        Sources {
            train: &[Path::new(train)],
            valid: Some(Path::new(valid)),
        },
        out,
        additional,
        snapshots,
    )
}

pub fn resume_sources(
    base: &Path,
    sources: Sources,
    out: &Path,
    additional: usize,
    snapshots: Option<&Path>,
) -> Result<()> {
    resume_with_topology(base, sources, out, additional, snapshots, None)
}

/// Change the replica layout only at an epoch boundary, keeping AdamW's effective batch intact.
pub fn resume_with_topology(
    base: &Path,
    sources: Sources,
    out: &Path,
    additional: usize,
    snapshots: Option<&Path>,
    layout: Option<(Topology, usize)>,
) -> Result<()> {
    ensure!(
        !out.exists() && additional > 0,
        "need fresh output and positive additional epochs"
    );
    let mut progress: Progress =
        serde_json::from_reader(fs::File::open(base.join("progress.json"))?)?;
    progress.schedule.validate()?;
    if let Some((new, micro)) = layout {
        let old_global = match progress.parallel {
            Some(t) => t.global_batch(progress.schedule.batch_size)?,
            None => progress.schedule.batch_size,
        };
        ensure!(
            new.global_batch(micro)? == old_global,
            "resume topology must preserve global batch {old_global}"
        );
        progress.parallel = Some(new);
        progress.schedule.batch_size = micro;
    }
    let (hashes, extra) = sources.fingerprints()?;
    ensure!(
        progress.corpus_hashes == hashes && progress.extra_train_hashes == extra,
        "training or validation corpus has changed"
    );
    if let Some(t) = progress.parallel {
        t.global_batch(progress.schedule.batch_size)?;
    }
    let scratch = tempfile::tempdir()?;
    let texts = TextSources::prepare(sources, scratch.path())?;
    let d = primary_device(progress.parallel);
    let model = Checkpoint::load(base, &d)?;
    let optimizer = AdamWConfig::new().init().load(base.join("optimizer.bpk"))?;
    fit(
        model, optimizer, progress, additional, &texts, out, snapshots, &d,
    )
}

pub fn pretrain(
    config: MetaIConfig,
    train: &str,
    valid: &str,
    tokenizer_path: Option<&Path>,
    output: &Path,
    schedule: &Schedule,
    snapshots: Option<&Path>,
) -> Result<()> {
    pretrain_sources(
        config,
        Sources {
            train: &[Path::new(train)],
            valid: Some(Path::new(valid)),
        },
        tokenizer_path,
        output,
        schedule,
        snapshots,
        None,
    )
}

fn primary_device(parallel: Option<Topology>) -> Device {
    match parallel {
        Some(t) => crate::backend::device_for_rank(0, t.world_size).autodiff(),
        None => get_device().autodiff(),
    }
}

pub fn pretrain_sources(
    mut config: MetaIConfig,
    sources: Sources,
    tokenizer_path: Option<&Path>,
    output: &Path,
    schedule: &Schedule,
    snapshots: Option<&Path>,
    parallel: Option<Topology>,
) -> Result<()> {
    if let Some(t) = parallel {
        t.global_batch(schedule.batch_size)?;
    }
    let (hashes, extra_train_hashes) = sources.fingerprints()?;
    config.validate()?;
    schedule.validate()?;
    ensure!(
        !output.exists(),
        "output already exists: {}",
        output.display()
    );
    ensure!(
        tokenizer_path.is_some() || config.vocab_size >= 260,
        "new byte-level tokenizer needs at least 260 entries"
    );
    let temp = tempfile::tempdir()?;
    let texts = TextSources::prepare(sources, temp.path())?;
    let tokenizer = match tokenizer_path {
        Some(p) => MetaITokenizer::new(p)?,
        None => MetaITokenizer::train(
            &texts.train,
            temp.path().join("tokenizer.json"),
            config.vocab_size,
        )?,
    };
    ensure!(
        tokenizer.vocab_size() <= config.vocab_size,
        "tokenizer exceeds model vocabulary"
    );
    config.vocab_size = tokenizer.vocab_size();
    let device = primary_device(parallel);
    device.seed(schedule.seed);
    let pad = tokenizer.pad_id().context("missing pad token")?;
    let model = MetaIModel::new(&config, pad, &device);
    let progress = Progress {
        epoch: 0,
        schedule: schedule.clone(),
        corpus_hashes: hashes,
        extra_train_hashes,
        parallel,
    };
    fit(
        Checkpoint {
            model,
            config,
            tokenizer,
        },
        AdamWConfig::new().init(),
        progress,
        schedule.num_epochs,
        &texts,
        output,
        snapshots,
        &device,
    )
}

#[allow(clippy::too_many_arguments)]
fn fit(
    mut ck: Checkpoint,
    mut optimizer: burn::optim::ModuleOptimizer,
    mut progress: Progress,
    epochs: usize,
    sources: &TextSources,
    output: &Path,
    snapshots: Option<&Path>,
    device: &Device,
) -> Result<()> {
    let schedule = progress.schedule.clone();
    let pad = ck.model.pad_id;
    let training = LazyTextDataset::from_files(
        &sources
            .train
            .iter()
            .map(|p| p.as_path())
            .collect::<Vec<_>>(),
        &ck.tokenizer,
        ck.config.max_seq_len,
    )?;
    let validation = sources
        .valid
        .as_deref()
        .map(|p| LazyTextDataset::from_files(&[p], &ck.tokenizer, ck.config.max_seq_len))
        .transpose()?;
    ensure!(
        validation.as_ref().is_none_or(|v| v.len() > 0),
        "no validation examples"
    );
    eprintln!(
        "runtime={:?} model_parameters={}",
        crate::backend::runtime(),
        ck.model.num_params()
    );
    let batcher = TextBatcher::new(device.clone(), pad);
    let save = |ck: &Checkpoint,
                path: &Path,
                opt: &burn::optim::ModuleOptimizer,
                progress: &Progress,
                report: Option<&serde_json::Value>| {
        ck.save_with(path, |dir| {
            opt.save(dir.join("optimizer.bpk"))?;
            if let Some(report) = report {
                fs::write(
                    dir.join("training-report.json"),
                    serde_json::to_vec_pretty(report)?,
                )?;
            }
            fs::write(
                dir.join("progress.json"),
                serde_json::to_vec_pretty(progress)?,
            )?;
            Ok(())
        })
    };
    for _ in 0..epochs {
        progress.epoch += 1;
        ck.model = if let Some(t) = progress.parallel {
            distributed::epoch(
                ck.model,
                &training,
                &schedule,
                progress.epoch,
                &mut optimizer,
                t,
            )?
        } else {
            schedule.epoch(
                ck.model,
                training.len(),
                progress.epoch,
                &mut optimizer,
                |model, indices| {
                    let b = batcher.batch(training.batch_rows(indices)?, device);
                    let mask = b.targets.clone().not_equal_elem(pad as i32).int();
                    Ok(masked_nll(
                        model.forward(b.inputs, None, None),
                        b.targets,
                        mask,
                    ))
                },
            )?
        };
        if let Some(dir) = snapshots {
            save(
                &ck,
                &dir.join(format!("epoch-{}", progress.epoch)),
                &optimizer,
                &progress,
                None,
            )?;
        }
    }
    let validation_report = if let Some(validation) = validation {
        let eval_device = device.clone().inner();
        let eval = ck.model.valid();
        let mut nll = 0.;
        let mut tokens = 0usize;
        for batch in (0..validation.len())
            .collect::<Vec<_>>()
            .chunks(schedule.batch_size)
        {
            let b = batcher.batch(validation.batch_rows(batch)?, &eval_device);
            let mask = b.targets.clone().not_equal_elem(pad as i32).float();
            tokens += batch.iter().map(|&i| validation.targets(i)).sum::<usize>();
            nll -= (token_logps(eval.forward(b.inputs, None, None), b.targets) * mask)
                .sum()
                .into_scalar::<f32>() as f64;
        }
        ensure!(tokens > 0 && nll.is_finite(), "invalid validation loss");
        println!("valid_tokens={tokens} valid_nll={:.6}", nll / tokens as f64);
        Some(serde_json::json!({"tokens": tokens, "nll": nll / tokens as f64}))
    } else {
        None
    };
    let pool = device.memory_pool_usage().map(|u| serde_json::json!({
        "bytes_in_use":u.bytes_in_use, "bytes_padding":u.bytes_padding, "bytes_reserved":u.bytes_reserved,
        "live_allocations":u.number_allocs
    }));
    let t = progress.parallel.unwrap_or(Topology {
        world_size: 1,
        accumulation_steps: 1,
    });
    let report = serde_json::json!({
        "runtime":format!("{:?}",crate::backend::runtime()), "parameters":ck.model.num_params(),
        "completed_epochs":progress.epoch, "world_size":t.world_size, "accumulation_steps":t.accumulation_steps,
        "micro_batch":schedule.batch_size, "global_batch":t.global_batch(schedule.batch_size)?,
        "training_blocks":training.len(), "train_targets_per_epoch":(0..training.len()).map(|i|training.targets(i)).sum::<usize>(),
        "validation":validation_report, "primary_allocator_before_save":pool
    });
    eprintln!("training_report={report}");
    save(&ck, output, &optimizer, &progress, Some(&report))
}
