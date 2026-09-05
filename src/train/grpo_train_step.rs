//! GRPO 强化学习训练：真正的 rollout（策略自回归采样）+ 规则奖励 + 组内基线
//!
//! 与旧实现的关键差异：
//! - 旧：直接拿 SFT 数据当 rollout，rewards 全 0，loss 恒为 KL 正则（模型学不到奖励信号）
//! - 新：每个 prompt 用策略采样 group_size 条响应 -> 规则奖励打分 -> 组内归一化优势
//!       -> 在采样序列上重算 policy/ref logprob -> GRPO loss -> 单次参数更新

use std::path::Path;

use burn::module::Module;
use burn::optim::{GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::backend::{get_device, MyAutodiffBackend};
use crate::data::{GRPODataset, MetaITokenizer};
use crate::model::{MetaIConfig, MetaIModel};
use crate::train::grpo::GRPOLoss;
use crate::train::grpo_rollout::{rollout_one, SamplingConfig};
use crate::train::reward::RewardKind;
use crate::train::MetaITrainingConfig;

/// 执行一步 GRPO：rollout -> reward -> loss（不更新参数）
#[allow(clippy::too_many_arguments)]
fn grpo_loss_step<B: AutodiffBackend>(
    policy: &MetaIModel<B>,
    reference: &MetaIModel<B>,
    tokenizer: &MetaITokenizer,
    dataset: &GRPODataset,
    chunk: &[usize],
    group_size: usize,
    max_new_tokens: usize,
    sampling: &SamplingConfig,
    reward_kind: RewardKind,
    pad_id: u32,
    eos_id: u32,
    device: &B::Device,
    grpo_loss: &GRPOLoss<B>,
    rng: &mut StdRng,
) -> anyhow::Result<Tensor<B, 1>> {
    let num_groups = chunk.len();
    let num_rows = num_groups * group_size;

    // ---- 1. 对每个 prompt rollout group_size 条响应 ----
    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(num_rows);
    let mut prompt_lens: Vec<usize> = Vec::with_capacity(num_rows);
    let mut row_lengths: Vec<usize> = Vec::with_capacity(num_rows);
    let mut reward_flat: Vec<f32> = Vec::with_capacity(num_rows);
    let mut row_instructions: Vec<String> = Vec::with_capacity(num_groups);

    for &idx in chunk {
        let item = &dataset.data[idx];
        row_instructions.push(item.instruction.clone());

        // rollout
        let mut responses: Vec<Vec<u32>> = Vec::with_capacity(group_size);
        for _ in 0..group_size {
            let resp = rollout_one(
                policy,
                device,
                &item.prompt,
                max_new_tokens,
                sampling,
                eos_id,
                rng,
            );
            responses.push(resp);
        }

        // 解码成文本做规则奖励
        let response_texts: Vec<String> = responses
            .iter()
            .map(|ids| tokenizer.decode(ids))
            .collect();
        let group_rewards: Vec<f32> = response_texts
            .iter()
            .map(|text| reward_kind.score(&item.instruction, text, 4, 512))
            .collect();

        for (resp, rew) in responses.iter().zip(group_rewards.iter()) {
            let mut row = item.prompt.clone();
            row.extend(resp.iter().copied());
            prompt_lens.push(item.prompt.len());
            row_lengths.push(row.len());
            rows.push(row);
            reward_flat.push(*rew);
        }
    }

    let max_len = *row_lengths.iter().max().unwrap_or(&1);
    if max_len <= 1 {
        anyhow::bail!("rollout 序列为空, 请检查数据/模型输出");
    }
    // 预测时输入长度 = max_len - 1（最后一个位置无需预测）
    let seq = max_len - 1;

    // ---- 2. 组装 pad 后的输入 / 目标 / mask ----
    // 注意：各 rollout 行长度不同，必须先把每行补 pad 到 max_len，
    // 否则短行会在 `row[t]`/`row[t+1]` 处索引越界。
    let rows_padded: Vec<Vec<u32>> = rows
        .iter()
        .map(|row| {
            let mut r = row.clone();
            r.resize(max_len, pad_id);
            r
        })
        .collect();
    let mut input_i32 = vec![pad_id as i32; num_rows * seq];
    let mut target_i32 = vec![pad_id as i32; num_rows * seq];
    let mut mask_i32 = vec![0i32; num_rows * seq];
    for (r, row) in rows_padded.iter().enumerate() {
        let plen = prompt_lens[r];
        let row_len = row_lengths[r];
        for t in 0..seq {
            // 位置 t 的输入 = row[t]，预测目标 = row[t+1]
            input_i32[r * seq + t] = row[t] as i32;
            target_i32[r * seq + t] = row[t + 1] as i32;
            // mask：目标为响应 token 且非 pad（t+1 >= plen 表示目标位于响应区）
            let is_response = (t + 1) >= plen && (t + 1) < row_len;
            if is_response {
                mask_i32[r * seq + t] = 1;
            }
        }
    }

    let input_t = Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(input_i32, [num_rows, seq]),
        device,
    );
    let target_t = Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(target_i32, [num_rows, seq]),
        device,
    );
    let mask_t = Tensor::<B, 3, Int>::from_data(
        burn::tensor::TensorData::new(mask_i32, [num_groups, group_size, seq]),
        device,
    );

    // ---- 3. Policy / Reference logprob（在采样序列上）----
    let gather = |model: &MetaIModel<B>| -> Tensor<B, 3> {
        let logits = model.forward(input_t.clone(), None, None);
        let logits = log_softmax(logits, 2); // [rows, seq, V]
        let target3 = target_t.clone().unsqueeze::<3>(); // [rows, seq, 1]
        logits
            .gather(2, target3)
            .squeeze::<3>() // [rows, seq]
            .reshape([num_groups, group_size, seq])
    };

    let policy_logps = gather(policy);
    let ref_logps = gather(reference).detach();

    let rewards = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(reward_flat, [num_groups, group_size]),
        device,
    );

    // ---- 4. GRPO loss ----
    Ok(grpo_loss.forward(policy_logps, ref_logps, rewards, mask_t))
}

/// GRPO 主入口：手写训练循环（Learner 的 wrapper 限制无法承载 rollout）
#[allow(clippy::too_many_arguments)]
pub fn run_grpo_training(
    data_path: &str,
    model_dir: &str,
    output_dir: &str,
    tcfg: &MetaITrainingConfig,
    grpo_cfg: &crate::train::grpo::GRPOConfig,
) -> anyhow::Result<()> {
    let device = get_device();
    let tokenizer = MetaITokenizer::new(&tcfg.tokenizer_path)?;
    let pad_id = tokenizer.pad_id().unwrap_or(0);
    let eos_id = tokenizer.eos_id().unwrap_or(2);

    let dataset = GRPODataset::from_file(data_path, &tokenizer, tcfg.model.max_seq_len)?;
    if dataset.data.is_empty() {
        anyhow::bail!("GRPO 数据集为空: {data_path}");
    }
    println!(
        "GRPO dataset: {} items, group_size={}, max_new_tokens={}",
        dataset.data.len(),
        grpo_cfg.group_size,
        grpo_cfg.max_new_tokens
    );
    if grpo_cfg.group_size < 2 {
        anyhow::bail!("GRPO group_size 必须 >= 2 (组内基线需要至少 2 条响应)");
    }

    // 加载 policy 与 reference（同一 SFT checkpoint）
    let model_cfg: MetaIConfig = tcfg.model.clone();
    let policy_init = MetaIModel::<MyAutodiffBackend>::new(&model_cfg, pad_id, &device);
    let mut policy = crate::train::load_model_checkpoint(policy_init, model_dir, &device);
    let reference = MetaIModel::<MyAutodiffBackend>::new(&model_cfg, pad_id, &device);
    let reference = crate::train::load_model_checkpoint(reference, model_dir, &device);

    // burn 0.19: AdamWConfig::init() 返回 OptimizerAdaptor<AdamW, M, B>，
    // M 由后续 step() 调用推断为 MetaIModel<MyAutodiffBackend>。
    let mut optimizer = tcfg.optimizer.init();
    let sampling = SamplingConfig {
        temperature: grpo_cfg.temperature,
        top_k: grpo_cfg.top_k,
        top_p: grpo_cfg.top_p,
    };
    let grpo_loss = GRPOLoss::<MyAutodiffBackend>::new(grpo_cfg.clone());

    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let checkpoint_dir = Path::new(output_dir).join("checkpoint");
    std::fs::create_dir_all(&checkpoint_dir)?;

    let batch_size = tcfg.batch_size.max(1);
    let mut global_step = 0usize;

    for epoch in 1..=tcfg.num_epochs {
        // 打乱数据顺序
        let mut order: Vec<usize> = (0..dataset.data.len()).collect();
        let mut rng = StdRng::seed_from_u64(tcfg.seed as u64 + epoch as u64 * 1_000_003);
        order.shuffle(&mut rng);

        let mut epoch_loss_sum = 0.0f64;
        let mut epoch_steps = 0usize;

        for chunk in order.chunks(batch_size) {
            let loss = grpo_loss_step(
                &policy,
                &reference,
                &tokenizer,
                &dataset,
                chunk,
                grpo_cfg.group_size,
                grpo_cfg.max_new_tokens,
                &sampling,
                grpo_cfg.reward,
                pad_id,
                eos_id,
                &device,
                &grpo_loss,
                &mut rng,
            )?;

            let loss_val: f32 = loss.clone().into_scalar();
            if !loss_val.is_finite() {
                eprintln!("[warn] step {global_step} 出现非有限 loss ({loss_val})，跳过更新");
                continue;
            }

            // 手动优化一步（只更新 policy）
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &policy);
            policy = optimizer.step(tcfg.learning_rate, policy, grads_params);

            epoch_loss_sum += loss_val as f64;
            epoch_steps += 1;
            global_step += 1;

            if global_step % 5 == 0 || global_step <= 3 {
                println!(
                    "[GRPO] epoch={epoch} step={global_step} loss={loss_val:.4}"
                );
            }
        }

        if epoch_steps > 0 {
            println!(
                "[GRPO] epoch {epoch} 完成, avg_loss={:.4}, steps={}",
                epoch_loss_sum / epoch_steps as f64,
                epoch_steps
            );
        }

        // 每个 epoch 保存 policy checkpoint（与其它训练产物兼容 model-N.bin）。
        // Module derive 自动实现 Clone，因此克隆一份用于按值消费的 save_file。
        let save_path = checkpoint_dir.join(format!("model-{}.bin", epoch));
        policy
            .clone()
            .save_file(&save_path, &recorder)
            .map_err(|e| anyhow::anyhow!("保存 GRPO checkpoint 失败: {e}"))?;
        println!("[GRPO] 已保存 checkpoint: {}", save_path.display());
    }

    Ok(())
}
