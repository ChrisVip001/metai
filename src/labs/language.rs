use super::*;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use rand::{rngs::StdRng, seq::SliceRandom, RngExt, SeedableRng};

const LEN: usize = 8;
const WIDTH: usize = 8;
fn sequence(first: i32) -> Vec<i32> {
    (0..LEN).map(|i| (first + i as i32) % 4).collect()
}
fn ids(rows: &[Vec<i32>], d: &Device) -> Tensor<2, Int> {
    let flat: Vec<_> = rows.iter().flatten().copied().collect();
    Tensor::<1, Int>::from_ints(flat.as_slice(), d).reshape([rows.len(), rows[0].len()])
}

#[derive(Module, Debug)]
struct LinearLm {
    embed: Embedding,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}
impl LinearLm {
    fn new(d: &Device) -> Self {
        let project = || LinearConfig::new(WIDTH, WIDTH).init(d);
        Self {
            embed: EmbeddingConfig::new(4, WIDTH).init(d),
            q: project(),
            k: project(),
            v: project(),
            out: LinearConfig::new(WIDTH, 4).init(d),
        }
    }
    /// S_t = S_(t-1) + k_t v_t^T, z_t = z_(t-1) + k_t. All features are positive.
    fn forward(&self, x: Tensor<2, Int>) -> Tensor<3> {
        let [n, l] = x.dims();
        let d = x.device();
        let h = self.embed.forward(x);
        let phi = |x| burn::tensor::activation::elu(x, 1.) + 1.;
        let q = phi(self.q.forward(h.clone()));
        let k = phi(self.k.forward(h.clone()));
        let v = self.v.forward(h.clone());
        let mut state = Tensor::zeros([n, WIDTH, WIDTH], &d);
        let mut norm = Tensor::zeros([n, 1, WIDTH], &d);
        let mut outputs = Vec::new();
        for i in 0..l {
            let row = |t: &Tensor<3>| t.clone().slice([0..n, i..i + 1]);
            let key = row(&k);
            state = state + key.clone().transpose().matmul(row(&v));
            norm = norm + key;
            let query = row(&q);
            let value =
                query.clone().matmul(state.clone()) / ((query * norm.clone()).sum_dim(2) + 1e-6);
            outputs.push(self.out.forward(value));
        }
        Tensor::cat(outputs, 1)
    }
}

#[derive(Module, Debug)]
struct Denoiser {
    embed: Embedding,
    hidden: Linear,
    out: Linear,
}
impl Denoiser {
    fn new(d: &Device) -> Self {
        Self {
            embed: EmbeddingConfig::new(5, WIDTH).init(d),
            hidden: LinearConfig::new(LEN * WIDTH + 1, 48).init(d),
            out: LinearConfig::new(48, LEN * 4).init(d),
        }
    }
    fn forward(&self, x: Tensor<2, Int>, t: Tensor<2>) -> Tensor<3> {
        let [n, _] = x.dims();
        let h = Tensor::cat(vec![self.embed.forward(x).reshape([n, LEN * WIDTH]), t], 1);
        self.out
            .forward(self.hidden.forward(h).tanh())
            .reshape([n, LEN, 4])
    }
    /// Keep the first symbol fixed and reveal one confident masked position per iteration.
    fn complete(&self, first: i32, d: &Device) -> Vec<i32> {
        let mut tokens = vec![4; LEN];
        tokens[0] = first;
        for remaining in (1..LEN).rev() {
            let t = Tensor::full([1, 1], remaining as f32 / (LEN - 1) as f32, d);
            let p =
                burn::tensor::activation::softmax(self.forward(ids(&[tokens.clone()], d), t), 2)
                    .into_data()
                    .iter::<f32>()
                    .collect::<Vec<_>>();
            let (j, token, _) = tokens
                .iter()
                .enumerate()
                .filter(|(_, v)| **v == 4)
                .flat_map(|(j, _)| (0..4).map(move |v| (j, v)))
                .map(|(j, v)| (j, v, p[j * 4 + v]))
                .max_by(|a, b| a.2.total_cmp(&b.2))
                .unwrap();
            tokens[j] = token as i32;
        }
        tokens
    }
}

pub(super) fn run(kind: Kind, s: &Schedule, path: &Path, d: &Device) -> Result<serde_json::Value> {
    let eval = d.clone().inner();
    let mut samples = Vec::new();
    if matches!(kind, Kind::Linear) {
        let m = s.fit(LinearLm::new(d), 32, |m, b| {
            let rows: Vec<_> = b.iter().map(|i| sequence((i % 4) as i32)).collect();
            let x = ids(&rows, d);
            let n = b.len();
            let logits = m.forward(x.clone().slice([0..n, 0..LEN - 1]));
            let labels = x.slice([0..n, 1..LEN]);
            Ok(-crate::train::token_logps(logits, labels).mean())
        })?;
        let m = roundtrip(m, path, LinearLm::new(&eval))?;
        for first in 0..4 {
            let mut row = vec![first];
            for i in 1..LEN {
                let next = m
                    .forward(ids(&[row.clone()], &eval))
                    .slice([0..1, i - 1..i])
                    .argmax(2)
                    .into_scalar::<i32>();
                row.push(next);
            }
            samples.push(row);
        }
    } else {
        let mut rng = StdRng::seed_from_u64(s.seed);
        let m = s.fit(Denoiser::new(d), 32, |m, b| {
            let mut noisy = Vec::new();
            let mut clean = Vec::new();
            let mut masks = Vec::new();
            let mut times = Vec::new();
            for &i in b {
                let x = sequence((i % 4) as i32);
                let mut y = x.clone();
                let mut mask = vec![0; LEN];
                let count = rng.random_range(1..LEN);
                let mut positions: Vec<_> = (1..LEN).collect();
                positions.shuffle(&mut rng);
                for &j in &positions[..count] {
                    y[j] = 4;
                    mask[j] = 1;
                }
                noisy.push(y);
                clean.push(x);
                masks.push(mask);
                times.push(count as f32 / (LEN - 1) as f32);
            }
            let z = m.forward(
                ids(&noisy, d),
                Tensor::<1>::from_floats(times.as_slice(), d).reshape([b.len(), 1]),
            );
            let mask = ids(&masks, d).float();
            let losses = -(crate::train::token_logps(z, ids(&clean, d)) * mask.clone()).sum_dim(1)
                / mask.sum_dim(1);
            Ok(losses.mean())
        })?;
        let m = roundtrip(m, path, Denoiser::new(&eval))?;
        samples = (0..4).map(|v| m.complete(v, &eval)).collect();
    }
    let correct = samples
        .iter()
        .enumerate()
        .filter(|(i, row)| **row == sequence(*i as i32))
        .count();
    Ok(serde_json::json!({"completions":samples,"exact_on_four_training_patterns":correct}))
}
