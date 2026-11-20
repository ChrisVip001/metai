//! Tool imitation and policy-gradient fine-tuning in a two-step, replayable environment.
use super::{ce, *};
use crate::train::{grpo::advantages, grpo_rollout::sample_token_from_logits, Checkpoint};
use burn::nn::{Linear, LinearConfig};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "action", deny_unknown_fields)]
enum Action {
    Lookup,
    Answer { bit: u8 },
}
const ACTIONS: [Action; 3] = [
    Action::Lookup,
    Action::Answer { bit: 0 },
    Action::Answer { bit: 1 },
];
#[derive(Clone)]
struct Episode {
    bit: u8,
    visible: bool,
    looked: bool,
    steps: usize,
    done: bool,
}
impl Episode {
    fn new(case: usize) -> Self {
        Self {
            bit: (case % 2) as u8,
            visible: case % 4 < 2,
            looked: false,
            steps: 0,
            done: false,
        }
    }
    fn state(&self) -> String {
        let value = if self.visible || self.looked {
            self.bit.to_string()
        } else {
            "?".into()
        };
        format!(
            "card {value} lookup {} answer",
            if self.looked { "used" } else { "ready" }
        )
    }
    fn oracle(&self) -> usize {
        if self.visible || self.looked {
            1 + self.bit as usize
        } else {
            0
        }
    }
    fn step(&mut self, action: Action) -> Result<f32> {
        ensure!(!self.done, "episode finished");
        self.steps += 1;
        match action {
            Action::Answer { bit } => {
                ensure!(bit < 2, "bit must be zero or one");
                self.done = true;
                Ok(if bit == self.bit { 1. } else { -1. })
            }
            Action::Lookup => {
                if self.looked || self.steps >= 2 {
                    self.done = true;
                    Ok(-1.)
                } else {
                    self.looked = true;
                    Ok(-0.15)
                }
            }
        }
    }
}

fn state_features(ck: &Checkpoint, text: &str, d: &Device) -> Result<Tensor<2>> {
    let ids = ck.tokenizer.encode(text);
    let n = ids.len();
    ensure!(n > 0 && n <= ck.config.max_seq_len, "state exceeds context");
    let x: Vec<_> = ids.iter().map(|&i| i as i32).collect();
    let hidden = ck.model.hidden(
        Tensor::<1, Int>::from_ints(x.as_slice(), d).reshape([1, n]),
        None,
        None,
    );
    Ok(hidden
        .reshape([n, ck.config.hidden_dim])
        .mean_dim(0)
        .detach())
}

pub fn run(base: &Path, path: &Path, s: &Schedule) -> Result<()> {
    let d = get_device().autodiff();
    d.seed(s.seed);
    let ck = Checkpoint::load(base, &d)?;
    let head = LinearConfig::new(ck.config.hidden_dim, 3).init(&d);
    // Cache the six distinct observations, including the two post-tool states.
    let mut table = std::collections::HashMap::new();
    let mut demos = Vec::new();
    for case in 0..4 {
        let mut env = Episode::new(case);
        while !env.done {
            let key = env.state();
            table.insert(key.clone(), state_features(&ck, &key, &d)?);
            let label = env.oracle();
            demos.push((key, label));
            env.step(ACTIONS[label])?;
        }
    }
    let head = s.fit(head, demos.len(), |policy, ids| {
        let x = Tensor::cat(ids.iter().map(|&i| table[&demos[i].0].clone()).collect(), 0);
        let y: Vec<_> = ids.iter().map(|&i| demos[i].1 as i32).collect();
        Ok(ce(policy.forward(x), &y, &d))
    })?;
    let imitation = head.clone();
    let mut rng = StdRng::seed_from_u64(s.seed);
    let head = s.fit(head, 4, |policy, ids| {
        let mut terms = Vec::new();
        for &case in ids {
            let mut returns = Vec::new();
            let mut trajectories = Vec::new();
            for _ in 0..6 {
                let mut env = Episode::new(case);
                let mut total = 0.;
                let mut logps = Vec::new();
                while !env.done {
                    let logits = policy.forward(table[&env.state()].clone());
                    let values = logits.clone().into_data().iter::<f32>().collect::<Vec<_>>();
                    let a = sample_token_from_logits(&values, 1., 0, 1., &mut rng) as usize;
                    logps.push(log_softmax(logits, 1).slice([0..1, a..a + 1]).reshape([1]));
                    total += env.step(ACTIONS[a])?;
                }
                returns.push(total);
                trajectories.push(Tensor::cat(logps, 0).sum());
            }
            for (lp, adv) in trajectories.into_iter().zip(advantages(&returns)) {
                terms.push(-lp * adv);
            }
        }
        Ok(Tensor::cat(terms, 0).mean())
    })?;
    ck.save_with(path, |dir| {
        head.clone().save_file(dir.join("agent.bpk"))?;
        imitation.save_file(dir.join("tool-sft.bpk"))?;
        let loaded = LinearConfig::new(ck.config.hidden_dim, 3)
            .init(&d)
            .try_load_file(dir.join("agent.bpk"))?;
        let mut episodes = Vec::new();
        for case in 0..4 {
            episodes.push(rollout(&ck, &loaded, case, &d)?);
        }
        let trace = serde_json::json!({"episodes":episodes,"max_steps":2,"seed":s.seed});
        std::fs::write(dir.join("traces.json"), serde_json::to_vec_pretty(&trace)?)?;
        println!("{trace}");
        Ok(())
    })
}

fn rollout(ck: &Checkpoint, head: &Linear, case: usize, d: &Device) -> Result<serde_json::Value> {
    ensure!(case < 4, "choose case 0 through 3");
    let mut env = Episode::new(case);
    let mut events = Vec::new();
    while !env.done {
        let state = env.state();
        let logits = head.forward(state_features(ck, &state, d)?);
        let action = ACTIONS[logits.argmax(1).into_scalar::<i32>() as usize];
        let reward = env.step(action)?;
        events.push(serde_json::json!({"state":state,"action":action,"reward":reward,"next_state":env.state(),"done":env.done}));
    }
    Ok(serde_json::json!(events))
}

pub fn play(dir: &Path, case: usize, imitation: bool) -> Result<serde_json::Value> {
    let device = get_device();
    let ck = Checkpoint::load(dir, &device)?;
    let path = dir.join(if imitation {
        "tool-sft.bpk"
    } else {
        "agent.bpk"
    });
    let head = LinearConfig::new(ck.config.hidden_dim, 3)
        .init(&device)
        .try_load_file(path)?;
    rollout(&ck, &head, case, &device)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lookup_changes_observation_and_is_bounded() {
        let mut a = Episode::new(2);
        let b = Episode::new(3);
        assert_eq!(a.state(), b.state());
        assert_eq!(a.step(Action::Lookup).unwrap(), -0.15);
        assert_ne!(a.state(), b.state());
        assert_eq!(a.step(Action::Lookup).unwrap(), -1.);
        assert!(a.step(Action::Lookup).is_err());
        assert!(serde_json::from_str::<Action>(r#"{"action":"launch"}"#).is_err());
    }
}
