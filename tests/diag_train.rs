#![recursion_limit = "1024"]
//! 诊断测试：检查 micro 模型的初始数值行为与崩溃 checkpoint 的退化情况
//! 运行: cargo test --release --test diag_train -- --nocapture diag_xxx

use burn::data::dataloader::batcher::Batcher;
use burn::module::Module;
use metai::backend::{get_device, MyBackend};
use metai::data::MetaITokenizer;
use metai::model::config::MetaIConfig;
use metai::model::MetaIModel;

fn stats_1d(t: &burn::tensor::Tensor<MyBackend, 1>, name: &str) {
    let t = t.clone();
    let mean = t.clone().mean().into_scalar();
    let min = t.clone().min().into_scalar();
    let max = t.clone().max().into_scalar();
    let nan_ratio = t.clone().is_nan().float().mean().into_scalar();
    let inf_ratio = t.clone().abs().equal_elem(f32::INFINITY).float().mean().into_scalar();
    println!(
        "[{}] mean={:.4} min={:.4} max={:.4} nan_ratio={:.6} inf_ratio={:.6}",
        name, mean, min, max, nan_ratio, inf_ratio
    );
}

fn build_batch(tokenizer: &MetaITokenizer, lines: &[String], start: usize) -> metai::data::data::TextBatch<MyBackend> {
    let config = MetaIConfig::micro();
    let items: Vec<Vec<u32>> = lines[start..start + 8]
        .iter()
        .map(|l| tokenizer.encode(l).into_iter().take(64).collect())
        .collect();
    let device = get_device();
    let batcher = metai::data::data::TextBatcher::<MyBackend>::new(device.clone(), 0);
    batcher.batch(items, &device)
}

#[test]
fn diag_initial_loss() {
    let device = get_device();
    let config = MetaIConfig::micro();
    let tokenizer = MetaITokenizer::new("tokenizer.json").expect("load tokenizer");
    let pad_id = tokenizer.pad_id().unwrap_or(0);

    let text = std::fs::read_to_string("/tmp/zh_train_s.txt").expect("read train data");
    let lines: Vec<String> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect();
    println!("lines: {}", lines.len());
    assert!(lines.len() >= 16);

    let batch = build_batch(&tokenizer, &lines, 0);
    let [bs, seq] = batch.inputs.dims();
    println!("batch dims: [{bs}, {seq}]");

    let model = MetaIModel::<MyBackend>::new(&config, pad_id, &device);
    let logits = model.forward(batch.inputs.clone(), None, None);
    let [bs2, seq2, vocab] = logits.dims();
    println!("logits dims: [{bs2}, {seq2}, {vocab}]");

    let logits_flat = logits.clone().flatten::<1>(0, 2);
    stats_1d(&logits_flat, "logits.flat");
    println!(
        "logits[0,0,0..16] = {:?}",
        logits
            .clone()
            .slice([0..1, 0..1, 0..16])
            .flatten::<1>(0, 2)
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>()
    );

    // 每行最大 logits：全遮罩行会表现为 -inf
    let row_max = logits.clone().max_dim(2).squeeze::<2>();
    let neg_inf_rows = row_max.clone().equal_elem(f32::NEG_INFINITY).float().mean().into_scalar();
    let nan_rows = row_max.clone().is_nan().float().mean().into_scalar();
    println!("rows with max=-inf: {neg_inf_rows:.6}, rows with max=NaN: {nan_rows:.6}");

    let logits_flat2 = logits.reshape([bs * seq, vocab]);
    let targets_flat = batch.targets.clone().reshape([bs * seq]);
    let loss = burn::nn::loss::CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![pad_id as usize]))
        .init(&device)
        .forward(logits_flat2, targets_flat);
    println!("initial CE loss (ignore pad): {:.4}", loss.into_scalar());

    // --- 阶段 2: 崩溃 checkpoint 分析 (放在同一测试避免并行编译问题) ---
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    let device = get_device();
    let config = MetaIConfig::micro();
    let tokenizer = MetaITokenizer::new("tokenizer.json").expect("load tokenizer");
    let pad_id = tokenizer.pad_id().unwrap_or(0);

    // 全新模型
    let fresh = MetaIModel::<MyBackend>::new(&config, pad_id, &device);
    let emb_fresh = fresh.embedding.weight.val().clone().flatten::<1>(0, 1);
    stats_1d(&emb_fresh, "fresh.embedding");

    // 加载崩溃 checkpoint
    let path = std::path::Path::new("/tmp/metai_diag/checkpoint/model-1.bin");
    if !path.exists() {
        println!("checkpoint not found, skip");
        return;
    }
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let crashed = MetaIModel::<MyBackend>::new(&config, pad_id, &device);
    let crashed = crashed
        .load_file(path.to_path_buf(), &recorder, &device)
        .expect("load crashed checkpoint");
    let emb_crash = crashed.embedding.weight.val().clone().flatten::<1>(0, 1);
    stats_1d(&emb_crash, "crashed.embedding");
    let out_w = crashed.output.weight.val().clone().flatten::<1>(0, 1);
    stats_1d(&out_w, "crashed.output.weight");

    let text = std::fs::read_to_string("/tmp/zh_train_s.txt").expect("read");
    let lines: Vec<String> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect();

    // 崩溃模型: 两个不同输入是否产生相同输出(退化)?
    let batch_a = build_batch(&tokenizer, &lines, 0);
    let batch_b = build_batch(&tokenizer, &lines, 8);
    let logits_a = crashed.forward(batch_a.inputs, None, None);
    let logits_b = crashed.forward(batch_b.inputs.clone(), None, None);
    let diff = (logits_a.clone() - logits_b.clone()).abs().mean().into_scalar();
    println!("crashed model: cross-input logits mean|diff| = {diff:.6}");
    stats_1d(&logits_a.clone().flatten::<1>(0, 2), "crashed.logits");
    println!(
        "crashed logits[0,0,0..32] = {:?}",
        logits_a
            .clone()
            .slice([0..1, 0..1, 0..32])
            .flatten::<1>(0, 2)
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>()
    );

    // 全新模型: 相同输入两次 forward 应一致；不同输入应有差异
    let f_a = fresh.forward(batch_b.inputs.clone(), None, None);
    let f_b = fresh.forward(batch_b.inputs, None, None);
    let f_same = (f_a.clone() - f_b.clone()).abs().mean().into_scalar();
    println!("fresh model: identical-input mean|diff| (应≈0) = {f_same:.6}");
    let f_diff = (logits_a - f_b).abs().mean().into_scalar();
    println!("fresh vs crashed on same input mean|diff| = {f_diff:.6}");
}
