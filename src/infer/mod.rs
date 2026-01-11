use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor};

use crate::data::MetaITokenizer;
use crate::model::{MetaIConfig, MetaIModel};

pub mod cache;

/// 文本生成器
///
/// 封装了模型和分词器，提供了便捷的文本生成接口。
/// 支持 KV Cache 加速、温度采样、Top-K 和 Top-P 采样策略。
pub struct Generator<B: Backend> {
    model: MetaIModel<B>,
    tokenizer: MetaITokenizer,
}

impl<B: Backend> Generator<B> {
    /// 创建新的生成器
    pub fn new(model: MetaIModel<B>, tokenizer: MetaITokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// 从检查点加载模型并创建生成器
    pub fn from_checkpoint(
        artifact_dir: &str,
        config: MetaIConfig,
        tokenizer: MetaITokenizer,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        use std::path::Path;

        let checkpoint_dir = Path::new(artifact_dir).join("checkpoint");
        if !checkpoint_dir.exists() {
            anyhow::bail!("Checkpoint directory not found: {:?}", checkpoint_dir);
        }

        // 查找最新的检查点
        let mut max_epoch = 0;
        if let Ok(entries) = std::fs::read_dir(&checkpoint_dir) {
            for entry in entries.flatten() {
                if let Some(file_name) = entry.file_name().to_str() {
                    if file_name.starts_with("model-") && file_name.ends_with(".bin") {
                        if let Some(epoch_str) = file_name
                            .strip_prefix("model-")
                            .and_then(|s| s.strip_suffix(".bin"))
                        {
                            if let Ok(epoch) = epoch_str.parse::<usize>() {
                                if epoch > max_epoch {
                                    max_epoch = epoch;
                                }
                            }
                        }
                    }
                }
            }
        }

        if max_epoch == 0 {
            anyhow::bail!("No checkpoint found in {:?}", checkpoint_dir);
        }

        let pad_id = tokenizer.pad_id().unwrap_or(0);
        let model = MetaIModel::new(&config, pad_id, device);

        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let model_path = checkpoint_dir.join(format!("model-{}.bin", max_epoch));

        // 加载模型权重
        let model = model
            .load_file(&model_path, &recorder, device)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

        Ok(Self { model, tokenizer })
    }

    /// 生成文本
    ///
    /// # 参数
    /// - `prompt`: 输入提示文本
    /// - `max_new_tokens`: 最大生成 token 数
    /// - `temperature`: 采样温度，0.0 表示贪婪采样，>0.0 表示随机采样
    /// - `top_k`: Top-K 采样，只从概率最高的 K 个 token 中采样
    /// - `top_p`: Top-P (Nucleus) 采样，只从累积概率达到 P 的 token 中采样
    ///
    /// # 返回
    /// 生成的完整文本（包含 prompt）
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> String {
        let device = self.model.embedding.weight.device();
        let tokens_ids = self.tokenizer.encode(prompt);
        let mut output_ids = tokens_ids.clone();

        // 1. 初始化 Cache
        let mut cache = crate::infer::cache::ModelCache::new(self.model.blocks.len());

        // 2. Prefill 阶段: 处理 Prompt
        // 将整个 Prompt 输入，填充 Cache，并获取最后一个 Token 的 logits
        let input = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(
                tokens_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                [1, tokens_ids.len()],
            ),
            &device,
        );
        let logits = self.model.forward(input, None, Some(&mut cache));

        // 提取最后一个 token 的 logits 用于预测下一个
        let [_batch, seq_len, vocab_size] = logits.dims();
        let mut next_token_logits = logits
            .slice([0..1, (seq_len - 1)..seq_len])
            .flatten::<1>(0, 2); // [1, Vocab]

        // 3. Decoding 循环
        for _ in 0..max_new_tokens {
            // --- 采样逻辑 ---
            // 应用温度
            if temperature > 0.0 {
                next_token_logits = next_token_logits / (temperature as f64);
            }

            // 获取概率分布
            let probs = burn::tensor::activation::softmax(next_token_logits, 0);

            // 采样 (Top-K, Top-P) - 逻辑保持不变
            let next_token_id = if temperature > 0.0 {
                let probs_vec: Vec<f32> = probs.clone().into_data().iter::<f32>().collect();
                let mut indexed_probs: Vec<(usize, f32)> =
                    probs_vec.into_iter().enumerate().collect();

                // Sort descending
                indexed_probs
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Top-K
                if top_k > 0 && top_k < vocab_size {
                    indexed_probs.truncate(top_k);
                }

                // Top-P (Nucleus)
                if top_p > 0.0 && top_p < 1.0 {
                    let mut cumulative_prob = 0.0;
                    let mut cutoff_idx = indexed_probs.len();
                    for (i, &(_, p)) in indexed_probs.iter().enumerate() {
                        cumulative_prob += p;
                        if cumulative_prob >= top_p {
                            cutoff_idx = i + 1;
                            break;
                        }
                    }
                    indexed_probs.truncate(cutoff_idx);
                }

                // Re-normalize and sample (Multinomial)
                let total_p: f32 = indexed_probs.iter().map(|&(_, p)| p).sum();
                let mut rng = rand::thread_rng();
                use rand::Rng;
                let r: f32 = rng.gen::<f32>() * total_p;
                let mut current_sum = 0.0;
                let mut sampled_id = indexed_probs[0].0;
                for (id, p) in indexed_probs {
                    current_sum += p;
                    if current_sum >= r {
                        sampled_id = id;
                        break;
                    }
                }
                sampled_id as u32
            } else {
                // 贪婪采样
                use burn::tensor::ElementConversion;
                probs.argmax(0).into_scalar().elem::<u32>()
            };

            output_ids.push(next_token_id);

            // 检查 EOS
            if Some(next_token_id) == self.tokenizer.pad_id() || next_token_id == 2 {
                break;
            }

            // --- 准备下一步输入 ---
            // 只输入刚刚生成的这一个 token
            let next_input = Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(vec![next_token_id as i32], [1, 1]),
                &device,
            );

            // Forward Pass with Cache
            // 输入: [1, 1], Cache: Included History. 输出: [1, 1, Vocab]
            let logits = self.model.forward(next_input, None, Some(&mut cache));

            // 直接获取 logits (seq_len=1)
            next_token_logits = logits.flatten::<1>(0, 2);
        }

        self.tokenizer.decode(&output_ids)
    }
}

/// 支持投机采样 (Speculative Decoding) 的生成器
pub struct SpeculativeGenerator<B: Backend> {
    draft_model: MetaIModel<B>,
    target_model: MetaIModel<B>,
    tokenizer: MetaITokenizer,
}

impl<B: Backend> SpeculativeGenerator<B> {
    pub fn new(
        draft_model: MetaIModel<B>,
        target_model: MetaIModel<B>,
        tokenizer: MetaITokenizer,
    ) -> Self {
        Self {
            draft_model,
            target_model,
            tokenizer,
        }
    }

    /// 应用投机采样生成文本
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        lookahead: usize, // K: 投机步数
        temperature: f32,
    ) -> String {
        let device = self.target_model.embedding.weight.device();
        let tokens_ids = self.tokenizer.encode(prompt);
        let mut output_ids = tokens_ids.clone();

        let mut draft_cache = cache::ModelCache::new(self.draft_model.blocks.len());
        let mut target_cache = cache::ModelCache::new(self.target_model.blocks.len());

        // 1. Prefill
        let input = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(
                tokens_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                [1, tokens_ids.len()],
            ),
            &device,
        );
        let _ = self
            .draft_model
            .forward(input.clone(), None, Some(&mut draft_cache));
        let target_logits = self
            .target_model
            .forward(input, None, Some(&mut target_cache));

        let mut next_token_logits = target_logits
            .slice([0..1, (tokens_ids.len() - 1)..tokens_ids.len()])
            .flatten::<1>(0, 2);

        let mut generated_count = 0;

        while generated_count < max_new_tokens {
            // --- 采样当前已知的最后一个 Token ---
            let probs = burn::tensor::activation::softmax(
                next_token_logits / (temperature as f64 + 1e-8),
                0,
            );
            let last_token_id = probs.argmax(0).into_scalar().elem::<u32>(); // 简化为贪婪

            if generated_count > 0 {
                output_ids.push(last_token_id);
                if Some(last_token_id) == self.tokenizer.pad_id() || last_token_id == 2 {
                    break;
                }
            }
            generated_count += 1;

            // --- Draft Step: 预测后续 lookahead 个 token ---
            let mut draft_tokens = Vec::new();
            let mut current_draft_token = last_token_id;

            for _ in 0..lookahead {
                let draft_input = Tensor::<B, 2, Int>::from_data(
                    burn::tensor::TensorData::new(vec![current_draft_token as i32], [1, 1]),
                    &device,
                );
                let d_logits = self
                    .draft_model
                    .forward(draft_input, None, Some(&mut draft_cache));
                current_draft_token = d_logits.argmax(2).into_scalar().elem::<u32>();
                draft_tokens.push(current_draft_token);
            }

            // --- Target Step: 一次性验证 Draft Tokens ---
            // 构建验证输入: [last_token, draft_token_1, ..., draft_token_k]
            let mut verify_vec = vec![last_token_id as i32];
            verify_vec.extend(draft_tokens.iter().map(|&t| t as i32));

            let verify_input = Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(verify_vec.clone(), [1, verify_vec.len()]),
                &device,
            );

            // 并行 Forward
            let v_logits = self
                .target_model
                .forward(verify_input, None, Some(&mut target_cache));

            // --- Verification & Rejection Sampling ---
            let mut accepted_count = 0;
            let mut final_next_logits = None;

            for i in 0..lookahead {
                // Target 对该位置的预测 (对应输入中的 verify_input[i])
                let target_pred_logits =
                    v_logits.clone().slice([0..1, i..i + 1]).flatten::<1>(0, 2);
                let target_pred_token = target_pred_logits
                    .clone()
                    .argmax(0)
                    .into_scalar()
                    .elem::<u32>();

                if target_pred_token == draft_tokens[i] {
                    accepted_count += 1;
                    output_ids.push(draft_tokens[i]);
                    generated_count += 1;

                    // 同步 Draft Cache (已经更新过了)
                } else {
                    // 拒绝！获取 Target 给出的正确下一个 Token 的 logits
                    final_next_logits = Some(target_pred_logits);
                    break;
                }
            }

            // 如果全部接受，还需要获取最后一个位置的 logits 以进行下一次循环
            if final_next_logits.is_none() {
                final_next_logits = Some(
                    v_logits
                        .slice([0..1, lookahead..lookahead + 1])
                        .flatten::<1>(0, 2),
                );
            }

            next_token_logits = final_next_logits.unwrap();

            // --- Cache Synchronization ---
            // Target Cache 此时已经增长了 verify_vec.len() = lookahead + 1
            // 实际上我们只接受了 accepted_count + 1 (最后一个是用来预测下一次的，还没存入 Cache)
            // 更新 Cache 到真正 accepted 的长度
            let total_valid_len = target_cache.seq_len() - (lookahead - accepted_count);
            target_cache.truncate(total_valid_len);
            draft_cache.truncate(total_valid_len);
        }

        self.tokenizer.decode(&output_ids)
    }
}

#[cfg(test)]
mod tests {}
