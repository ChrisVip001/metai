//! Collect a batch on-policy, then make two updates on those same frozen trajectories.
use super::{
    grpo::{advantages, policy_loss, Algorithm},
    grpo_rollout::{rollout_one, SamplingConfig},
    reward::RewardKind,
    *,
};
use crate::{
    backend::get_device,
    data::{sft::SFTBatcher, GRPODataset},
};
use burn::{data::dataloader::batcher::Batcher, nn::LinearConfig};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

/// Compare the frozen teacher's likelihoods of two verdicts, including multi-token verdicts.
fn judge(ck: &Checkpoint, question: &str, candidate: &str, device: &Device) -> Result<f32> {
    let prompt = ck.tokenizer.encode(&format!(
        "Question: {question}\nCandidate: {candidate}\nCorrect?"
    ));
    let verdicts = [" yes", " no"];
    let mut scores = [0.; 2];
    for (index, verdict) in verdicts.iter().enumerate() {
        let answer = ck.tokenizer.encode(verdict);
        ensure!(
            !answer.is_empty() && prompt.len() + answer.len() <= ck.config.max_seq_len,
            "judge context exhausted"
        );
        let mut ids = prompt.clone();
        ids.extend(&answer);
        let mut mask = vec![0; prompt.len()];
        mask.extend(vec![1; answer.len()]);
        let b = SFTBatcher::new(device.clone(), ck.model.pad_id).batch(vec![(ids, mask)], device);
        scores[index] = (token_logps(ck.model.forward(b.inputs, None, None), b.targets)
            * b.mask.float())
        .sum()
        .into_scalar::<f32>();
    }
    Ok(1. / (1. + (scores[1] - scores[0]).exp()))
}

#[derive(clap::Args)]
pub struct RolloutConfig {
    #[arg(long, default_value_t = 4)]
    pub group_size: usize,
    #[arg(long, default_value_t = 8)]
    pub max_new_tokens: usize,
    #[arg(long, value_enum, default_value = "rule")]
    pub reward: RewardKind,
    /// Optional frozen yes/no judge checkpoint, for local AI feedback.
    #[arg(long)]
    pub judge_dir: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = 0.2)]
    pub clip: f64,
    #[arg(long, default_value_t = 0.02)]
    pub beta: f64,
}

pub fn run(
    data: &str,
    dir: &Path,
    output: &Path,
    schedule: &Schedule,
    method: Algorithm,
    cfg: &RolloutConfig,
) -> Result<()> {
    schedule.validate()?;
    ensure!(
        cfg.group_size >= 2
            && cfg.max_new_tokens > 0
            && cfg.clip > 0.
            && cfg.clip < 1.
            && cfg.beta.is_finite()
            && cfg.beta >= 0.,
        "invalid rollout parameters"
    );
    let device = get_device().autodiff();
    device.seed(schedule.seed);
    let mut ck = Checkpoint::load(dir, &device)?;
    let ds = GRPODataset::from_file(data, &ck.tokenizer, ck.config.max_seq_len)?;
    ensure!(!ds.data.is_empty() && ds.data.iter().all(|r|r.prompt.len().saturating_add(cfg.max_new_tokens)<=ck.config.max_seq_len), "empty dataset or rollout exceeds context");
    let judge_device = device.clone().inner();
    let teacher = cfg
        .judge_dir
        .as_ref()
        .map(|p| Checkpoint::load(p, &judge_device))
        .transpose()?;
    println!(
        "feedback={}",
        if teacher.is_some() {
            "frozen_yes_no_judge"
        } else {
            "rule"
        }
    );
    let reference = ck.model.clone().no_grad();
    let mut critic = LinearConfig::new(ck.config.hidden_dim, 1).init(&device);
    let mut critic_opt = AdamWConfig::new().init();
    let mut optimizer = AdamWConfig::new().init();
    let eos = ck.tokenizer.eos_id().context("missing eos")?;
    let batcher = SFTBatcher::new(device.clone(), ck.model.pad_id);
    let sampling = SamplingConfig {
        temperature: 1.,
        top_k: 0,
        top_p: 1.,
    };
    let mut rng = StdRng::seed_from_u64(schedule.seed);
    let mut order: Vec<_> = (0..ds.data.len()).collect();
    for epoch in 1..=schedule.num_epochs {
        order.shuffle(&mut rng);
        for indices in order.chunks(schedule.batch_size) {
            let old = ck.model.clone().no_grad();
            let mut sequences = Vec::new();
            let mut reward_values = Vec::new();
            let mut group_adv = Vec::new();
            for &i in indices {
                let row = &ds.data[i];
                let mut group = Vec::new();
                for _ in 0..cfg.group_size {
                    let response = rollout_one(
                        &old,
                        &device,
                        &row.prompt,
                        cfg.max_new_tokens,
                        &sampling,
                        eos,
                        &mut rng,
                    );
                    let answer = ck.tokenizer.decode(&response);
                    let score = match &teacher {
                        Some(j) => judge(j, &row.instruction, &answer, &judge_device)?,
                        None => cfg.reward.score(&row.answer, &answer, 4, 512),
                    };
                    group.push(score);
                    let mut mask = vec![0; row.prompt.len()];
                    mask.extend(vec![1; response.len()]);
                    let mut ids = row.prompt.clone();
                    ids.extend(response);
                    sequences.push((ids, mask));
                }
                group_adv.extend(advantages(&group));
                reward_values.extend(group);
            }
            let b = batcher.batch(sequences, &device);
            let [n, s] = b.inputs.dims();
            let logps = |model: &MetaIModel| {
                token_logps(
                    model.forward(b.inputs.clone(), None, None),
                    b.targets.clone(),
                )
            };
            let old_lp = logps(&old).detach();
            let ref_lp = logps(&reference).detach();
            let rewards = Tensor::<1>::from_floats(reward_values.as_slice(), &device)
                .reshape([n, 1])
                .expand([n, s]);
            let mask = b.mask.float();
            // Terminal reward, gamma=lambda=1. The baseline sees only the prefix of each action.
            let adv = if method == Algorithm::Ppo {
                let h = old.hidden(b.inputs.clone(), None, None).detach();
                let v = critic.forward(h).reshape([n, s]);
                let advantage = rewards - v;
                let loss = (advantage.clone().square() * mask.clone()).sum() / mask.clone().sum();
                let grad = GradientsParams::from_grads(loss.backward(), &critic);
                critic = critic_opt.step(schedule.learning_rate, critic, grad);
                advantage.detach()
            } else {
                Tensor::<1>::from_floats(group_adv.as_slice(), &device)
                    .reshape([n, 1])
                    .expand([n, s])
            };
            for _ in 0..2 {
                let loss = policy_loss(
                    method,
                    logps(&ck.model),
                    old_lp.clone(),
                    ref_lp.clone(),
                    adv.clone(),
                    mask.clone(),
                    cfg.clip,
                    cfg.beta,
                );
                ensure!(
                    loss.clone().into_scalar::<f32>().is_finite(),
                    "nonfinite policy loss"
                );
                let grads = GradientsParams::from_grads(loss.backward(), &ck.model);
                ck.model = optimizer.step(schedule.learning_rate, ck.model, grads);
            }
            println!(
                "epoch={epoch} method={method:?} rollouts={n} mean_reward={:.4}",
                reward_values.iter().sum::<f32>() / n as f32
            );
        }
    }
    ck.save(output)
}
