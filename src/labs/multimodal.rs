//! Project image rows and optional waveform frames into the embedding space of a frozen LM.
use super::*;
use crate::{
    data::{chat_prompt, MetaITokenizer},
    train::{masked_nll, Checkpoint},
};
use burn::nn::{Linear, LinearConfig};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};

#[derive(Deserialize)]
struct Sample {
    image: [f32; 16],
    #[serde(default)]
    audio: Option<[f32; 16]>,
    question: String,
    #[serde(default)]
    answer: Option<String>,
}
#[derive(Module, Debug)]
struct Bridge {
    vision: Linear,
    sound: Option<Linear>,
}
impl Bridge {
    fn new(width: usize, audio: bool, d: &Device) -> Self {
        let layer = || LinearConfig::new(4, width).init(d);
        Self {
            vision: layer(),
            sound: audio.then(layer),
        }
    }
    fn prefix(&self, row: &Sample, d: &Device) -> Result<Tensor<3>> {
        let frames = |x: &[f32; 16]| Tensor::<1>::from_floats(x.as_slice(), d).reshape([1, 4, 4]);
        ensure!(row.image.iter().all(|x| x.is_finite()), "invalid pixels");
        let vision = self.vision.forward(frames(&row.image));
        match (&self.sound, &row.audio) {
            (None, None) => Ok(vision),
            (Some(sound), Some(wave)) => {
                ensure!(wave.iter().all(|x| x.is_finite()), "invalid audio samples");
                Ok(Tensor::cat(vec![vision, sound.forward(frames(wave))], 1))
            }
            _ => anyhow::bail!("audio presence must match the bridge configuration"),
        }
    }
}

/// Residual vocabulary logits from the current text state and all sensory frames.
#[derive(Module, Debug)]
struct Readout {
    image: Linear,
    audio: Option<Linear>,
    text: Linear,
    vocabulary: Linear,
}
impl Readout {
    fn new(width: usize, vocab: usize, audio: bool, d: &Device) -> Self {
        let projection = || LinearConfig::new(4 * width, width * 2).init(d);
        Self {
            image: projection(),
            audio: audio.then(projection),
            text: LinearConfig::new(width, width * 2).init(d),
            vocabulary: LinearConfig::new(width * 2, vocab).init(d),
        }
    }
    fn forward(&self, prefix: Tensor<3>, text: Tensor<3>) -> Tensor<3> {
        let [batch, frames, width] = prefix.dims();
        let image = self
            .image
            .forward(
                prefix
                    .clone()
                    .slice([0..batch, 0..4])
                    .reshape([batch, 1, 4 * width]),
            );
        let condition = if let Some(audio) = &self.audio {
            let sound =
                audio.forward(
                    prefix
                        .slice([0..batch, 4..frames])
                        .reshape([batch, 1, 4 * width]),
                );
            // A learned low-rank interaction can distinguish pairs with identical single-modality marginals.
            image.clone() * sound.clone() + image + sound
        } else {
            image
        };
        self.vocabulary.forward(burn::tensor::activation::relu(
            self.text.forward(text) + condition,
        ))
    }
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum AdapterConfig {
    // Historical prefix-only checkpoints contain a bare audio flag.
    Prefix(bool),
    Configured { audio: bool, readout: bool },
}

#[derive(Module, Debug)]
struct Adapter {
    bridge: Bridge,
    readout: Option<Readout>,
}

fn logits(
    ck: &Checkpoint,
    prefix: Tensor<3>,
    tokens: &[u32],
    readout: Option<&Readout>,
    d: &Device,
) -> Result<Tensor<3>> {
    let prefix_len = prefix.dims()[1];
    let n = tokens.len();
    ensure!(
        n > 0 && prefix_len + n <= ck.config.max_seq_len,
        "modality prefixes and text exceed context"
    );
    let ids: Vec<_> = tokens.iter().map(|&x| x as i32).collect();
    let text = ck
        .model
        .embedding
        .forward(Tensor::<1, Int>::from_ints(ids.as_slice(), d).reshape([1, n]));
    let hidden = ck.model.hidden_embeddings(
        Tensor::cat(vec![prefix.clone(), text], 1),
        Tensor::<2>::zeros([1, prefix_len + n], d).bool(),
        None,
    );
    let text = hidden.slice([0..1, prefix_len..prefix_len + n]);
    let base = ck.model.output.forward(text.clone());
    Ok(match readout {
        Some(adapter) => base + adapter.forward(prefix, text),
        None => base,
    })
}

fn training_tokens(
    tokenizer: &MetaITokenizer,
    question: &str,
    answer: &str,
    continuation: bool,
) -> Result<(usize, Vec<u32>)> {
    let template = chat_prompt(question, "");
    let mut ids = tokenizer.encode(&template);
    let prompt = ids.len();
    if continuation {
        let full = tokenizer.encode(&(template + answer));
        ensure!(
            full.starts_with(&ids) && full.len() > prompt,
            "tokenizer cannot keep the prompt boundary for this answer"
        );
        ids = full;
        ids.extend(tokenizer.eos_id());
    } else {
        ids.extend(tokenizer.response(answer));
    }
    Ok((prompt, ids))
}

pub fn train(base: &Path, data: &Path, out: &Path, audio: bool, s: &Schedule) -> Result<()> {
    train_with_options(base, data, out, audio, false, false, s)
}

pub fn train_with_options(
    base: &Path,
    data: &Path,
    out: &Path,
    audio: bool,
    train_readout: bool,
    continuation_targets: bool,
    s: &Schedule,
) -> Result<()> {
    let d = get_device().autodiff();
    d.seed(s.seed);
    let mut ck = Checkpoint::load(base, &d)?;
    ck.model = ck.model.no_grad();
    let mut rows = Vec::new();
    for line in BufReader::new(std::fs::File::open(data)?).lines() {
        let row: Sample = serde_json::from_str(&line?)?;
        ensure!(
            !row.question.trim().is_empty()
                && row.answer.as_ref().is_some_and(|a| !a.trim().is_empty()),
            "need a question and answer"
        );
        let (prompt, ids) = training_tokens(
            &ck.tokenizer,
            &row.question,
            row.answer.as_deref().unwrap(),
            continuation_targets,
        )?;
        rows.push((row, prompt, ids));
    }
    let adapter = s.fit(
        Adapter {
            bridge: Bridge::new(ck.config.hidden_dim, audio, &d),
            readout: train_readout
                .then(|| Readout::new(ck.config.hidden_dim, ck.config.vocab_size, audio, &d)),
        },
        rows.len(),
        |m, indices| {
            let mut losses = Vec::new();
            for &i in indices {
                let (row, prompt, ids) = &rows[i];
                let n = ids.len() - 1;
                let z = logits(
                    &ck,
                    m.bridge.prefix(row, &d)?,
                    &ids[..n],
                    m.readout.as_ref(),
                    &d,
                )?;
                let targets: Vec<_> = ids[1..].iter().map(|&x| x as i32).collect();
                let mask: Vec<_> = (1..=n).map(|t| i32::from(t >= *prompt)).collect();
                let tensor = |x: &[i32]| Tensor::<1, Int>::from_ints(x, &d).reshape([1, n]);
                losses.push(masked_nll(z, tensor(&targets), tensor(&mask)));
            }
            Ok(Tensor::cat(losses, 0).mean())
        },
    )?;
    ck.save_with(out, |p| {
        adapter.bridge.save_file(p.join("bridge.bpk"))?;
        if let Some(readout) = adapter.readout {
            readout.save_file(p.join("readout.bpk"))?;
        }
        let config = if train_readout {
            AdapterConfig::Configured {
                audio,
                readout: true,
            }
        } else {
            AdapterConfig::Prefix(audio)
        };
        std::fs::write(p.join("audio.json"), serde_json::to_vec(&config)?)?;
        Ok(())
    })
}

pub fn generate(dir: &Path, data: &Path, budget: usize) -> Result<()> {
    let d = get_device();
    let ck = Checkpoint::load(dir, &d)?;
    let config = serde_json::from_reader(std::fs::File::open(dir.join("audio.json"))?)?;
    let (audio, has_readout) = match config {
        AdapterConfig::Prefix(audio) => (audio, false),
        AdapterConfig::Configured { audio, readout } => (audio, readout),
    };
    let bridge =
        Bridge::new(ck.config.hidden_dim, audio, &d).try_load_file(dir.join("bridge.bpk"))?;
    let readout_path = dir.join("readout.bpk");
    let readout = if has_readout {
        Some(
            Readout::new(ck.config.hidden_dim, ck.config.vocab_size, audio, &d)
                .try_load_file(readout_path)?,
        )
    } else {
        None
    };
    for line in BufReader::new(std::fs::File::open(data)?).lines() {
        let row: Sample = serde_json::from_str(&line?)?;
        let prefix = bridge.prefix(&row, &d)?;
        let mut ids = ck.tokenizer.encode(&chat_prompt(&row.question, ""));
        let start = ids.len();
        ensure!(
            prefix.dims()[1]
                .saturating_add(ids.len())
                .saturating_add(budget)
                <= ck.config.max_seq_len,
            "generation exceeds context"
        );
        for _ in 0..budget {
            let n = ids.len();
            let z = logits(&ck, prefix.clone(), &ids, readout.as_ref(), &d)?;
            let next = z.slice([0..1, n - 1..n]).argmax(2).into_scalar::<u32>();
            ids.push(next);
            if Some(next) == ck.tokenizer.eos_id() {
                break;
            }
        }
        println!(
            "{}",
            serde_json::json!({"question":row.question,"answer":ck.tokenizer.decode(&ids[start..]),"token_ids":&ids[start..]})
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetaIConfig, MetaIModel};
    #[test]
    fn continuation_decodes_without_inventing_or_trimming_answer_whitespace() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let text = temp.path().join("corpus");
        std::fs::write(&text, "red is a color.\n")?;
        let tokenizer = MetaITokenizer::train(&[text], temp.path().join("bpe.json"), 270)?;
        for answer in ["yes", "no", " yes"] {
            let (start, ids) = training_tokens(&tokenizer, "Match?", answer, true)?;
            assert_eq!(ids.last().copied(), tokenizer.eos_id());
            assert_eq!(tokenizer.decode(&ids[start..]), answer);
        }
        Ok(())
    }

    #[test]
    fn response_loss_backpropagates_to_both_prefix_encoders() {
        let d = get_device().autodiff();
        let config = MetaIConfig::tiny();
        let frozen = MetaIModel::new(&config, 0, &d).no_grad();
        let bridge = Bridge::new(config.hidden_dim, true, &d);
        let row = Sample {
            image: [1.; 16],
            audio: Some([0.3; 16]),
            question: "match?".into(),
            answer: None,
        };
        let prefix = bridge.prefix(&row, &d).unwrap();
        let text = frozen
            .embedding
            .forward(Tensor::<2, Int>::from_ints([[4, 5]], &d));
        let h = frozen.hidden_embeddings(
            Tensor::cat(vec![prefix, text], 1),
            Tensor::<2>::zeros([1, 10], &d).bool(),
            None,
        );
        let loss = -log_softmax(frozen.output.forward(h), 2)
            .slice([0..1, 9..10, 7..8])
            .sum();
        let grads = loss.backward();
        for weight in [
            &bridge.vision.weight,
            &bridge.sound.as_ref().unwrap().weight,
        ] {
            let g = weight
                .val()
                .grad(&grads)
                .unwrap()
                .abs()
                .max()
                .into_scalar::<f32>();
            assert!(g.is_finite() && g > 0.);
        }
    }
}
