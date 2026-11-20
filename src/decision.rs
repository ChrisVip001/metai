//! A fixed-schema Jev-style classifier with independent calibration and test splits.
use crate::{
    backend::get_device,
    train::{Checkpoint, Schedule},
};
use anyhow::{ensure, Context, Result};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::log_softmax, Device, Int, Tensor},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeSet, HashSet},
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

#[derive(Deserialize)]
struct Example {
    text: String,
    label: String,
}
#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    // Checkpoints written before the pooling field used the final token.
    #[default]
    Last,
    Mean,
}

#[derive(Serialize, Deserialize)]
struct Schema {
    labels: Vec<String>,
    temperature: f64,
    #[serde(default)]
    pooling: Pooling,
}

fn read(path: &Path, seen: &mut HashSet<String>) -> Result<Vec<Example>> {
    let mut rows = Vec::new();
    for (i, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let row: Example = serde_json::from_str(&line?)
            .with_context(|| format!("{} line {}", path.display(), i + 1))?;
        let normalized = row.text.split_whitespace().collect::<Vec<_>>().join(" ");
        ensure!(
            !normalized.is_empty() && !row.label.trim().is_empty(),
            "empty decision text or label"
        );
        ensure!(
            seen.insert(normalized),
            "duplicate decision input in data splits"
        );
        rows.push(row);
    }
    ensure!(!rows.is_empty(), "empty decision split");
    Ok(rows)
}

fn feature(ck: &Checkpoint, text: &str, pooling: Pooling, device: &Device) -> Result<Tensor<2>> {
    ensure!(!text.trim().is_empty(), "empty input");
    let ids: Vec<i32> = ck
        .tokenizer
        .encode(text)
        .into_iter()
        .map(|i| i as i32)
        .collect();
    let n = ids.len();
    ensure!(
        n > 0 && n <= ck.config.max_seq_len,
        "decision text exceeds context or tokenizes to empty"
    );
    let input = Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([1, n]);
    let hidden = ck.model.hidden(input, None, None);
    let pooled = match pooling {
        Pooling::Last => hidden.slice([0..1, n - 1..n]),
        Pooling::Mean => hidden.mean_dim(1),
    };
    Ok(pooled.reshape([1, ck.config.hidden_dim]).detach())
}

fn probabilities(row: &[f64], t: f64) -> Vec<f64> {
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = row.iter().map(|x| ((x - max) / t).exp()).collect();
    let z: f64 = weights.iter().sum();
    weights.into_iter().map(|x| x / z).collect()
}

fn metrics(rows: &[(Vec<f64>, usize)], t: f64, threshold: f64) -> serde_json::Value {
    let (mut correct, mut accepted, mut errors) = (0, 0, 0);
    let (mut nll, mut brier) = (0., 0.);
    let classes = rows[0].0.len();
    let mut confusion = vec![vec![0usize; classes]; classes];
    let mut bins = [(0usize, 0., 0usize); 5];
    for (z, y) in rows {
        let p = probabilities(z, t);
        let (k, &conf) = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        confusion[*y][k] += 1;
        correct += usize::from(k == *y);
        nll -= p[*y].max(f64::MIN_POSITIVE).ln();
        brier += p
            .iter()
            .enumerate()
            .map(|(i, x)| (x - f64::from(i == *y)).powi(2))
            .sum::<f64>();
        if conf >= threshold {
            accepted += 1;
            errors += usize::from(k != *y);
        }
        let bin = &mut bins[((conf * 5.) as usize).min(4)];
        bin.0 += 1;
        bin.1 += conf;
        bin.2 += usize::from(k == *y);
    }
    let n = rows.len() as f64;
    let ece = bins
        .iter()
        .map(|(_, c, a)| (c - *a as f64).abs())
        .sum::<f64>()
        / n;
    serde_json::json!({"n":rows.len(),"accuracy":correct as f64/n,"nll":nll/n,"confusion_true_by_predicted":confusion,"brier_sum_classes":brier/n,"ece5":ece,
        "bins_count_confidence_sum_correct":bins,"threshold":threshold,"coverage":accepted as f64/n,
        "selective_risk":(accepted>0).then(||errors as f64/accepted as f64)})
}

pub fn train(
    base: &Path,
    train: &Path,
    calibration: &Path,
    test: &Path,
    output: &Path,
    schedule: &Schedule,
) -> Result<()> {
    train_with_pooling(
        base,
        train,
        calibration,
        test,
        output,
        schedule,
        Pooling::Last,
    )
}

pub fn train_with_pooling(
    base: &Path,
    train: &Path,
    calibration: &Path,
    test: &Path,
    output: &Path,
    schedule: &Schedule,
    pooling: Pooling,
) -> Result<()> {
    let device = get_device().autodiff();
    device.seed(schedule.seed);
    let ck = Checkpoint::load(base, &device)?;
    let mut seen = HashSet::new();
    let training = read(train, &mut seen)?;
    let calibration = read(calibration, &mut seen)?;
    let test = read(test, &mut seen)?;
    let labels: Vec<_> = training
        .iter()
        .map(|r| r.label.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    ensure!(labels.len() >= 2, "need two training labels");
    let encode = |rows: &[Example]| -> Result<Vec<(Tensor<2>, usize)>> {
        rows.iter()
            .map(|r| {
                Ok((
                    feature(&ck, &r.text, pooling, &device)?,
                    labels
                        .binary_search(&r.label)
                        .ok()
                        .context("unknown label")?,
                ))
            })
            .collect()
    };
    let features = encode(&training)?;
    let head = LinearConfig::new(ck.config.hidden_dim, labels.len()).init(&device);
    let head = schedule.fit(head, features.len(), |h, ids| {
        let x = Tensor::cat(ids.iter().map(|&i| features[i].0.clone()).collect(), 0);
        let y: Vec<_> = ids.iter().map(|&i| features[i].1 as i32).collect();
        Ok(-log_softmax(h.forward(x), 1)
            .gather(
                1,
                Tensor::<1, Int>::from_ints(y.as_slice(), &device).reshape([ids.len(), 1]),
            )
            .mean())
    })?;
    let score = |rows: &[Example]| -> Result<Vec<(Vec<f64>, usize)>> {
        Ok(encode(rows)?
            .into_iter()
            .map(|(x, y)| (head.forward(x).into_data().iter::<f64>().collect(), y))
            .collect())
    };
    let calibration = score(&calibration)?;
    let temperature = (-12..=12)
        .map(|i| 2f64.powf(i as f64 / 4.))
        .min_by(|a, b| {
            let error = |t| {
                calibration
                    .iter()
                    .map(|(z, y)| -probabilities(z, t)[*y].max(f64::MIN_POSITIVE).ln())
                    .sum::<f64>()
            };
            error(*a).total_cmp(&error(*b))
        })
        .unwrap();
    let heldout = score(&test)?;
    let report = serde_json::json!({"train":training.len(),"calibration":calibration.len(),"test_before":metrics(&heldout,1.,0.8),
        "test_after":metrics(&heldout,temperature,0.8),"temperature":temperature});
    println!("{report}");
    ck.save_with(output, |dir| {
        head.save_file(dir.join("decision.bpk"))?;
        std::fs::write(
            dir.join("schema.json"),
            serde_json::to_vec_pretty(&Schema {
                labels,
                temperature,
                pooling,
            })?,
        )?;
        std::fs::write(
            dir.join("evaluation.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        Ok(())
    })
}

pub fn predict(dir: &Path, text: &str) -> Result<serde_json::Value> {
    let device = get_device();
    let ck = Checkpoint::load(dir, &device)?;
    let schema: Schema = serde_json::from_reader(File::open(dir.join("schema.json"))?)?;
    ensure!(
        !schema.labels.is_empty() && schema.temperature.is_finite() && schema.temperature > 0.,
        "invalid decision schema"
    );
    let head: Linear = LinearConfig::new(ck.config.hidden_dim, schema.labels.len())
        .init(&device)
        .try_load_file(dir.join("decision.bpk"))?;
    let z = head
        .forward(feature(&ck, text, schema.pooling, &device)?)
        .into_data()
        .iter::<f64>()
        .collect::<Vec<_>>();
    let p = probabilities(&z, schema.temperature);
    let i = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
    Ok(
        serde_json::json!({"label":schema.labels[i],"labels":schema.labels,"probabilities":p,"confidence":p[i]}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_schema_preserves_final_token_pooling() {
        let schema: Schema =
            serde_json::from_str(r#"{"labels":["blue","red"],"temperature":8.0}"#).unwrap();
        assert!(matches!(schema.pooling, Pooling::Last));
    }
}
