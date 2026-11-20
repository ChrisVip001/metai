//! Public CLI contracts and numerical invariants for the CPU course.
use anyhow::Result;
use burn::tensor::{Int, Tensor};
use metai::{
    backend::get_device,
    train::{
        grpo::{advantages, policy_loss, Algorithm},
        Checkpoint,
    },
    AttentionVariant, MetaIConfig, MetaIModel, MlaConfig,
};
use std::{path::Path, process::Command};

fn cli(root: &Path, args: &[&str]) -> String {
    let result = Command::new(env!("CARGO_BIN_EXE_metai"))
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout).unwrap()
}

#[test]
fn pretrain_resume_and_posttrain_checkpoints() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    std::fs::write(
        root.join("train"),
        "red is red\nblue is blue\none plus one\ntwo plus two\n",
    )?;
    std::fs::write(root.join("valid"), "red and blue\n")?;
    std::fs::write(root.join("sft"), include_str!("../examples/tiny/sft.jsonl"))?;
    std::fs::write(
        root.join("pairs"),
        include_str!("../examples/tiny/preferences.jsonl"),
    )?;
    cli(
        root,
        &[
            "pretrain",
            "--train-path",
            "train",
            "--valid-path",
            "valid",
            "--output-dir",
            "base",
            "--checkpoint-dir",
            "checkpoints",
            "--num-epochs",
            "2",
            "--batch-size",
            "2",
        ],
    );
    cli(
        root,
        &[
            "resume-pretrain",
            "--train-path",
            "train",
            "--valid-path",
            "valid",
            "--model-dir",
            "checkpoints/epoch-1",
            "--output-dir",
            "resumed",
        ],
    );
    let d = get_device();
    let logits = |name: &str| -> Result<Tensor<3>> {
        let ck = Checkpoint::load(&root.join(name), &d)?;
        Ok(ck
            .model
            .forward(Tensor::<2, Int>::from_ints([[4, 5, 6]], &d), None, None))
    };
    let error = (logits("base")? - logits("resumed")?)
        .abs()
        .max()
        .into_scalar::<f32>();
    assert!(error < 1e-5, "epoch resume differs: {error}");
    cli(
        root,
        &[
            "train-sft",
            "--data-path",
            "sft",
            "--model-dir",
            "base",
            "--output-dir",
            "sft-model",
        ],
    );
    // Batch evaluation reports only generated IDs and their literal decoded text.
    let questions = r#"{"id":"held-out","instruction":"which color?","input":"blue"}"#;
    std::fs::write(root.join("questions.jsonl"), questions)?;
    let response: serde_json::Value = serde_json::from_str(&cli(
        root,
        &[
            "eval-chat",
            "--model-dir",
            "sft-model",
            "--questions",
            "questions.jsonl",
            "--max-len",
            "3",
        ],
    ))?;
    assert_eq!(response["id"], "held-out");
    let tokens: Vec<u32> = serde_json::from_value(response["token_ids"].clone())?;
    let tokenizer = metai::MetaITokenizer::new(root.join("sft-model/tokenizer.json"))?;
    assert!((1..=3).contains(&tokens.len()));
    assert_eq!(response["answer"], tokenizer.decode(&tokens));
    let mut prompt = tokenizer.encode(&metai::data::chat_prompt("which color?", "blue"));
    prompt.extend(tokens);
    let full = cli(
        root,
        &[
            "generate",
            "--model-dir",
            "sft-model",
            "--chat",
            "--prompt",
            "which color?\nblue",
            "--max-len",
            "3",
        ],
    );
    assert_eq!(full, format!("{}\n", tokenizer.decode(&prompt)));
    let before = logits("sft-model")?;
    for (cmd, data) in [
        ("train-lora", "sft"),
        ("train-dpo", "pairs"),
        ("train-grpo", "sft"),
        ("train-gspo", "sft"),
        ("train-ppo", "sft"),
        ("train-cispo", "sft"),
        ("distill", "train"),
        ("train-thinking", "sft"),
    ] {
        let mut args = vec![
            cmd,
            "--data-path",
            data,
            "--model-dir",
            "sft-model",
            "--output-dir",
            cmd,
        ];
        if matches!(
            cmd,
            "train-grpo" | "train-gspo" | "train-ppo" | "train-cispo"
        ) {
            args.extend(["--max-new-tokens", "2"]);
        }
        cli(root, &args);
        let z = logits(cmd)?;
        assert_eq!(z.dims(), before.dims());
        assert!(z.abs().max().into_scalar::<f32>().is_finite());
        if matches!(cmd, "train-dpo" | "train-lora") {
            assert!(
                (logits(cmd)? - before.clone())
                    .abs()
                    .max()
                    .into_scalar::<f32>()
                    > 1e-6
            );
        }
    }
    let result = cli(
        root,
        &[
            "think",
            "--model-dir",
            "train-thinking",
            "--prompt",
            "red",
            "--max-len",
            "2",
            "--budget",
            "1",
        ],
    );
    let result: serde_json::Value = serde_json::from_str(&result)?;
    assert!(result["latent_steps_per_token"]
        .as_array()
        .unwrap()
        .iter()
        .all(|n| n == 1));
    let greedy = cli(
        root,
        &[
            "generate",
            "--model-dir",
            "base",
            "--prompt",
            "red",
            "--max-len",
            "4",
        ],
    );
    let spec = cli(
        root,
        &[
            "generate-spec",
            "--model-dir",
            "base",
            "--draft-dir",
            "distill",
            "--prompt",
            "red",
            "--max-len",
            "4",
            "--lookahead",
            "3",
        ],
    );
    assert_eq!(greedy, spec);
    assert_eq!(
        (logits("sft-model")? - before)
            .abs()
            .max()
            .into_scalar::<f32>(),
        0.
    );
    Ok(())
}

#[test]
fn clipped_cispo_keeps_gradient_and_masks_padding() {
    let d = get_device().autodiff();
    assert_eq!(advantages(&[2., 2.]), vec![0., 0.]);
    for (method, expected) in [
        (Algorithm::Ppo, 0.),
        (Algorithm::Grpo, 0.),
        (Algorithm::Cispo, -1.2),
    ] {
        let current = Tensor::<2>::from_floats([[-0.1, -0.3]], &d).require_grad();
        let loss = policy_loss(
            method,
            current.clone(),
            Tensor::from_floats([[-2., -0.3]], &d),
            current.clone().detach(),
            Tensor::ones([1, 2], &d),
            Tensor::from_floats([[1., 0.]], &d),
            0.2,
            0.,
        );
        let grad = current
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();
        assert!(
            (grad[0] - expected).abs() < 1e-5 && grad[1] == 0.,
            "{method:?}: {grad:?}"
        );
    }
}

#[test]
fn cached_chunks_and_padding_match_full_sequence() {
    let d = get_device();
    for attn in [
        AttentionVariant::Gqa,
        AttentionVariant::Mla(MlaConfig::new(8, 4)),
    ] {
        let mut cfg = MetaIConfig::tiny();
        cfg.attn = attn;
        cfg.num_layers = 2;
        let model = MetaIModel::new(&cfg, 0, &d);
        let x = Tensor::<2, Int>::from_ints([[0, 4, 5, 6, 7]], &d);
        let full = model.forward(x.clone(), None, None);
        let mut cache = metai::infer::cache::ModelCache::new(2);
        let a = model.forward(x.clone().slice([0..1, 0..3]), None, Some(&mut cache));
        let b = model.forward(x.slice([0..1, 3..5]), None, Some(&mut cache));
        let diff = (Tensor::cat(vec![a, b], 1) - full)
            .abs()
            .max()
            .into_scalar::<f32>();
        assert!(diff < 1e-5, "{attn:?}: {diff}");
    }
}

#[test]
fn response_mask_includes_first_answer_and_excludes_padding() {
    use burn::data::dataloader::batcher::Batcher;
    use metai::data::sft::SFTBatcher;
    let d = get_device();
    let batch = SFTBatcher::new(d.clone(), 0).batch(
        vec![
            (vec![5, 6, 7, 2], vec![0, 0, 1, 1]),
            (vec![9, 8], vec![0, 1]),
        ],
        &d,
    );
    assert_eq!(
        batch.targets.into_data().iter::<i32>().collect::<Vec<_>>(),
        [6, 7, 2, 8, 0, 0]
    );
    assert_eq!(
        batch.mask.into_data().iter::<i32>().collect::<Vec<_>>(),
        [0, 1, 1, 1, 0, 0]
    );
}

#[test]
fn top_one_gate_receives_a_nonzero_gradient() {
    let d = get_device().autodiff();
    for router in [
        metai::RouterVariant::SoftmaxTopK,
        metai::RouterVariant::SigmoidTopK,
    ] {
        let mut cfg = MetaIConfig::tiny();
        cfg.active_experts = 1;
        cfg.router = router;
        let m = MetaIModel::new(&cfg, 0, &d);
        let logits = m.forward(Tensor::<2, Int>::from_ints([[5, 6, 7]], &d), None, None);
        let grad = m.blocks[0]
            .mlp
            .gate
            .weight
            .val()
            .grad(&logits.square().sum().backward())
            .unwrap();
        let max = grad.abs().max().into_scalar::<f32>();
        assert!(max.is_finite() && max > 1e-8);
    }
}

#[test]
fn gspo_clips_per_response_and_ignores_pad_and_length() {
    let d = get_device().autodiff();
    let log2 = 2.0_f32.ln();
    // Ratios 2 and 1/2 have geometric mean 1. The short negative response hits the lower clip.
    let current = Tensor::<2>::from_floats([[log2, -log2], [-log2, 8.]], &d).require_grad();
    let old = Tensor::<2>::zeros([2, 2], &d);
    let a = Tensor::<2>::from_floats([[1., 1.], [-1., -1.]], &d);
    let mask = Tensor::<2>::from_floats([[1., 1.], [1., 0.]], &d);
    let loss = policy_loss(
        Algorithm::Gspo,
        current.clone(),
        old.clone(),
        old,
        a,
        mask,
        0.2,
        0.,
    );
    assert!((loss.clone().into_scalar::<f32>() + 0.1).abs() < 1e-6);
    let grads = current.grad(&loss.backward()).unwrap().into_data();
    let g = grads.as_slice::<f32>().unwrap();
    assert!((g[0] + 0.25).abs() < 1e-6 && (g[1] + 0.25).abs() < 1e-6);
    assert_eq!(&g[2..], &[0., 0.]);
}
