use anyhow::{ensure, Result};
use burn::{
    data::dataset::Dataset,
    tensor::{Int, Tensor},
};
use metai::{
    backend::get_device,
    data::data::{LazyTextDataset, TextDataset},
    train::Checkpoint,
    MetaITokenizer,
};
use std::{fs, path::Path, process::Command};

fn run(root: &Path, args: &[&str]) -> Result<()> {
    let out = Command::new(env!("CARGO_BIN_EXE_metai"))
        .current_dir(root)
        .args(args)
        .output()?;
    ensure!(
        out.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Ok(())
}

#[test]
fn two_rank_accumulation_matches_global_batch_and_resume() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let root = dir.path();
    let zh = [
        "你好世界。",
        "红色的花。",
        "数一数今天的花。",
        "明天见。",
        "树是绿的。",
    ];
    let en = [
        "red",
        "one two three",
        "sky",
        "the wind is cool",
        "many many blue flowers",
    ];
    for (name, lines) in [("zh.jsonl", zh), ("en.jsonl", en)] {
        let json: Vec<_> = lines
            .iter()
            .map(|t| serde_json::json!({"text":t,"source":"excluded-metadata"}).to_string())
            .collect();
        fs::write(root.join(name), json.join("\n"))?;
    }
    fs::write(root.join("all"), [zh.join("\n"), en.join("\n")].join("\n"))?;
    fs::write(root.join("valid"), "validation has other words\n")?;
    run(
        root,
        &[
            "train-cluster",
            "--world-size",
            "1",
            "--size",
            "tiny",
            "--train-path",
            "all",
            "--valid-path",
            "valid",
            "--output-dir",
            "ref",
            "--num-epochs",
            "2",
            "--batch-size",
            "8",
        ],
    )?;
    run(
        root,
        &[
            "train-cluster",
            "--size",
            "tiny",
            "--world-size",
            "2",
            "--batch-size",
            "2",
            "--accumulation-steps",
            "2",
            "--chinese-path",
            "zh.jsonl",
            "--english-path",
            "en.jsonl",
            "--valid-path",
            "valid",
            "--num-epochs",
            "2",
            "--checkpoint-dir",
            "snap",
            "--output-dir",
            "parallel",
            "--tokenizer-path",
            "ref/tokenizer.json",
        ],
    )?;
    run(
        root,
        &[
            "resume-pretrain",
            "--model-dir",
            "snap/epoch-1",
            "--train-path",
            "zh.jsonl",
            "--train-path",
            "en.jsonl",
            "--valid-path",
            "valid",
            "--output-dir",
            "resumed",
        ],
    )?;
    let device = get_device();
    let logits = |name: &str| -> Result<Tensor<3>> {
        let ck = Checkpoint::load(&root.join(name), &device)?;
        Ok(ck.model.forward(
            Tensor::<2, Int>::from_ints([[4, 12, 19, 21]], &device),
            None,
            None,
        ))
    };
    for name in ["ref", "resumed"] {
        let delta = (logits("parallel")? - logits(name)?)
            .abs()
            .max()
            .into_scalar::<f32>();
        eprintln!("two_rank_vs_{name}_max_abs_logits={delta}");
        assert!(delta < 1e-4, "{name} differs by {delta}");
    }
    // Both paths now start from the exact same single-device weights AND AdamW moments.
    run(
        root,
        &[
            "pretrain",
            "--train-path",
            "all",
            "--valid-path",
            "valid",
            "--batch-size",
            "8",
            "--output-dir",
            "single",
        ],
    )?;
    let mut resume_args = vec![
        "resume-pretrain",
        "--model-dir",
        "single",
        "--train-path",
        "all",
        "--valid-path",
        "valid",
        "--output-dir",
        "single-next",
    ];
    run(root, &resume_args)?;
    resume_args.pop();
    resume_args.extend([
        "resharded",
        "--world-size",
        "2",
        "--batch-size",
        "2",
        "--accumulation-steps",
        "2",
    ]);
    run(root, &resume_args)?;
    let delta = (logits("single-next")? - logits("resharded")?)
        .abs()
        .max()
        .into_scalar::<f32>();
    eprintln!("single_to_two_rank_resume_max_abs_logits={delta}");
    assert!(delta < 1e-4);
    let metadata: serde_json::Value =
        serde_json::from_reader(fs::File::open(root.join("resharded/training-report.json"))?)?;
    assert_eq!(metadata["global_batch"], 8);
    assert_eq!(metadata["world_size"], 2);
    assert_eq!(metadata["completed_epochs"], 2);
    fs::write(root.join("en.jsonl"), r#"{"text":"changed"}"#)?;
    assert!(run(
        root,
        &[
            "resume-pretrain",
            "--model-dir",
            "snap/epoch-1",
            "--train-path",
            "zh.jsonl",
            "--train-path",
            "en.jsonl",
            "--valid-path",
            "valid",
            "--output-dir",
            "bad"
        ]
    )
    .is_err());
    assert!(!root.join("bad").exists());
    Ok(())
}

#[test]
fn disk_index_preserves_every_chunk_and_strips_line_endings() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let text = dir.path().join("text");
    fs::write(&text, "abcd efgh ijkl mnop\r\n\r\nqrst uvwx yz.\n")?;
    let tok = MetaITokenizer::train(std::slice::from_ref(&text), dir.path().join("bpe"), 260)?;
    let path = text.to_str().unwrap();
    let eager = TextDataset::from_file(path, &tok, 5)?;
    let lazy = LazyTextDataset::new(path, &tok, 5)?;
    assert!(eager.len() > 2);
    assert_eq!(eager.len(), lazy.len());
    for i in 0..eager.len() {
        assert_eq!(eager.get(i)?, lazy.get(i)?);
    }
    Ok(())
}
