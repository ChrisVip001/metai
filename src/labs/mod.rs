//! Small synthetic labs. Each command trains, saves, reloads and evaluates on CPU by default.
pub mod agent;
mod language;
pub mod multimodal;
mod sensory;
use crate::{backend::get_device, train::Schedule};
use anyhow::{ensure, Result};
use burn::{
    module::Module,
    tensor::{activation::log_softmax, Device, Int, Tensor},
};
use std::path::Path;

#[derive(Clone, Copy, clap::ValueEnum, Debug)]
pub enum Kind {
    Vision,
    Omni,
    Linear,
    Diffusion,
}

fn ce(logits: Tensor<2>, targets: &[i32], device: &Device) -> Tensor<1> {
    let n = targets.len();
    -log_softmax(logits, 1)
        .gather(
            1,
            Tensor::<1, Int>::from_ints(targets, device).reshape([n, 1]),
        )
        .mean()
}

fn roundtrip<M: Module>(model: M, path: &Path, fresh: M) -> Result<M> {
    model.save_file(path)?;
    Ok(fresh.try_load_file(path)?)
}

pub fn run(kind: Kind, path: &Path, schedule: &Schedule) -> Result<()> {
    ensure!(!path.exists(), "output already exists");
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let tmp = tempfile::tempdir_in(parent)?;
    let model_path = tmp.path().join("model.bpk");
    let device = get_device().autodiff();
    device.seed(schedule.seed);
    let report = match kind {
        Kind::Vision | Kind::Omni => sensory::run(kind, schedule, &model_path, &device)?,
        Kind::Linear | Kind::Diffusion => language::run(kind, schedule, &model_path, &device)?,
    };
    let report =
        serde_json::json!({"lab":format!("{kind:?}"),"seed":schedule.seed,"result":report});
    std::fs::write(
        tmp.path().join("evaluation.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    std::fs::rename(tmp.path(), path)?;
    println!("{report}");
    Ok(())
}
