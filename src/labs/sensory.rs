use super::*;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use rand::{rngs::StdRng, RngExt, SeedableRng};

/// Image is a 4x4 grayscale stripe. Audio is a 16-sample waveform at 16 Hz.
/// The prompt selects the image bit, audio bit, or whether both agree.
struct Scene {
    image: Vec<f32>,
    audio: Vec<f32>,
    query: i32,
    answer: i32,
}
fn scenes(n: usize, seed: u64, omni: bool) -> Vec<Scene> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let visual = (i % 2) as i32;
            let tone = ((i / 2) % 2) as i32;
            let query = if omni { ((i / 4) % 3) as i32 } else { 0 };
            let image = (0..16)
                .map(|j| {
                    let stripe = if visual == 0 { j % 4 < 2 } else { j / 4 < 2 };
                    f32::from(stripe) + rng.random_range(-0.12..0.12)
                })
                .collect();
            let audio = (0..16)
                .map(|j| {
                    (std::f32::consts::TAU * (tone + 1) as f32 * j as f32 / 16.).sin()
                        + rng.random_range(-0.1..0.1)
                })
                .collect();
            let answer = match query {
                0 => visual,
                1 => tone,
                _ => i32::from(visual == tone),
            };
            Scene {
                image,
                audio,
                query,
                answer,
            }
        })
        .collect()
}

#[derive(Module, Debug)]
struct SensorModel {
    image: Linear,
    audio: Option<Linear>,
    instruction: Embedding,
    fuse: Linear,
    answer: Linear,
    audio_out: Option<Linear>,
    image_out: Option<Linear>,
}
impl SensorModel {
    fn new(omni: bool, d: &Device) -> Self {
        let linear = |a, b| LinearConfig::new(a, b).init(d);
        Self {
            image: linear(16, 12),
            audio: omni.then(|| linear(16, 12)),
            instruction: EmbeddingConfig::new(3, 12).init(d),
            fuse: linear(if omni { 36 } else { 24 }, 32),
            answer: linear(32, 2),
            audio_out: omni.then(|| linear(32, 16)),
            image_out: omni.then(|| linear(32, 16)),
        }
    }
    fn forward(&self, batch: &[&Scene], d: &Device) -> (Tensor<2>, Tensor<1>) {
        let n = batch.len();
        let matrix = |v: Vec<f32>| Tensor::<1>::from_floats(v.as_slice(), d).reshape([n, 16]);
        let image = matrix(batch.iter().flat_map(|s| s.image.iter().copied()).collect());
        let query: Vec<_> = batch.iter().map(|s| s.query).collect();
        let mut parts = vec![
            self.image.forward(image.clone()),
            self.instruction
                .forward(Tensor::<1, Int>::from_ints(query.as_slice(), d).reshape([n, 1]))
                .reshape([n, 12]),
        ];
        let audio = matrix(batch.iter().flat_map(|s| s.audio.iter().copied()).collect());
        if let Some(encoder) = &self.audio {
            parts.push(encoder.forward(audio.clone()));
        }
        let h = self.fuse.forward(Tensor::cat(parts, 1)).tanh();
        let recon = match (&self.image_out, &self.audio_out) {
            (Some(im), Some(au)) => {
                (im.forward(h.clone()) - image).square().mean()
                    + (au.forward(h.clone()) - audio).square().mean()
            }
            _ => Tensor::zeros([1], d),
        };
        (self.answer.forward(h), recon)
    }
}
pub(super) fn run(kind: Kind, s: &Schedule, path: &Path, d: &Device) -> Result<serde_json::Value> {
    let omni = matches!(kind, Kind::Omni);
    let data = scenes(48, s.seed, omni);
    let m = s.fit(SensorModel::new(omni, d), data.len(), |m, ids| {
        let batch: Vec<_> = ids.iter().map(|&i| &data[i]).collect();
        let labels: Vec<_> = batch.iter().map(|s| s.answer).collect();
        let (logits, recon) = m.forward(&batch, d);
        Ok(ce(logits, &labels, d) + recon * 0.1)
    })?;
    let eval = d.clone().inner();
    let m = roundtrip(m, path, SensorModel::new(omni, &eval))?;
    let test = scenes(24, s.seed.wrapping_add(1), omni);
    let (z, recon) = m.forward(&test.iter().collect::<Vec<_>>(), &eval);
    let predicted: Vec<_> = z.argmax(1).into_data().iter::<i32>().collect();
    let correct = predicted
        .iter()
        .zip(&test)
        .filter(|(p, s)| **p == s.answer)
        .count();
    Ok(
        serde_json::json!({"test_scenes":test.len(),"correct":correct,"predictions":predicted,"reconstruction_mse_sum":recon.into_scalar::<f32>()}),
    )
}
