use burn::module::Module;
use burn::tensor::{Int, Tensor};

use crate::data::MetaITokenizer;
use crate::model::{MetaIConfig, MetaIModel};

pub mod cache;

/// 从工件目录加载最新 epoch 的模型权重（与 `Generator::from_checkpoint` 共用）。
///
/// `config` 必须与训练时一致（Burn checkpoint 只保存权重，不保存结构）。
pub fn load_model_from_checkpoint(
    artifact_dir: &str,
    config: MetaIConfig,
    pad_id: u32,
    device: &burn::tensor::Device,
) -> anyhow::Result<MetaIModel> {
    MetaIModel::new(&config, pad_id, device)
        .try_load_file(std::path::Path::new(artifact_dir).join("model.bpk"))
        .map_err(anyhow::Error::from)
}

/// 文本生成器
///
/// 封装了模型和分词器，提供了便捷的文本生成接口。
/// 支持 KV Cache 加速、温度采样、Top-K 和 Top-P 采样策略。
pub struct Generator {
    model: MetaIModel,
    tokenizer: MetaITokenizer,
}

impl Generator {
    /// 创建新的生成器
    pub fn new(model: MetaIModel, tokenizer: MetaITokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// 从检查点加载模型并创建生成器
    pub fn from_checkpoint(
        artifact_dir: &str,
        config: MetaIConfig,
        tokenizer: MetaITokenizer,
        device: &burn::tensor::Device,
    ) -> anyhow::Result<Self> {
        let pad_id = tokenizer.pad_id().unwrap_or(0);
        let model = load_model_from_checkpoint(artifact_dir, config, pad_id, device)?;
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
        let mut ids = self.tokenizer.encode(prompt);
        if ids.is_empty() || max_new_tokens == 0 {
            return prompt.to_owned();
        }
        let suffix = self.completion_ids(&ids, max_new_tokens, temperature, top_k, top_p);
        ids.extend(suffix);
        self.tokenizer.decode(&ids)
    }

    /// Generated tokens only, including the stop token when one was emitted.
    pub fn completion_ids(
        &self,
        tokens_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Vec<u32> {
        let device = self.model.embedding.weight.device();
        let mut output_ids = Vec::new();
        if tokens_ids.is_empty() || max_new_tokens == 0 {
            return output_ids;
        }

        // 1. 初始化 Cache
        let mut cache = crate::infer::cache::ModelCache::new(self.model.blocks.len());

        // 2. Prefill 阶段: 处理 Prompt
        // 将整个 Prompt 输入，填充 Cache，并获取最后一个 Token 的 logits
        let input = Tensor::<2, Int>::from_data(
            burn::tensor::TensorData::new(
                tokens_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                [1, tokens_ids.len()],
            ),
            &device,
        );
        let logits = self.model.forward(input, None, Some(&mut cache));

        // 提取最后一个 token 的 logits 用于预测下一个
        let [_batch, seq_len, _vocab_size] = logits.dims();
        let mut next_token_logits = logits
            .slice([0..1, (seq_len - 1)..seq_len])
            .flatten::<1>(0, 2); // [1, Vocab]

        // 3. Decoding 循环
        for _ in 0..max_new_tokens {
            let next_token_id = if temperature > 0.0 {
                let row: Vec<f32> = next_token_logits.into_data().iter().collect();
                crate::train::grpo_rollout::sample_token_from_logits(
                    &row,
                    temperature,
                    top_k,
                    top_p,
                    &mut rand::rng(),
                )
            } else {
                next_token_logits.argmax(0).into_scalar::<u32>()
            };

            output_ids.push(next_token_id);

            // 检查 EOS
            if Some(next_token_id) == self.tokenizer.pad_id()
                || Some(next_token_id) == self.tokenizer.eos_id()
                || output_ids.len() == max_new_tokens
            {
                break;
            }

            // --- 准备下一步输入 ---
            // 只输入刚刚生成的这一个 token
            let next_input = Tensor::<2, Int>::from_data(
                burn::tensor::TensorData::new(vec![next_token_id as i32], [1, 1]),
                &device,
            );

            // Forward Pass with Cache
            // 输入: [1, 1], Cache: Included History. 输出: [1, 1, Vocab]
            let logits = self.model.forward(next_input, None, Some(&mut cache));

            // 直接获取 logits (seq_len=1)
            next_token_logits = logits.flatten::<1>(0, 2);
        }

        output_ids
    }
}

/// 支持投机采样 (Speculative Decoding) 的生成器
pub struct SpeculativeGenerator {
    draft_model: MetaIModel,
    target_model: MetaIModel,
    tokenizer: MetaITokenizer,
}

impl SpeculativeGenerator {
    pub fn new(
        draft_model: MetaIModel,
        target_model: MetaIModel,
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
        _temperature: f32,
    ) -> String {
        let device = self.target_model.embedding.weight.device();
        let mut context = self.tokenizer.encode(prompt);
        if context.is_empty() || lookahead == 0 {
            return prompt.to_owned();
        }
        let end = context.len().saturating_add(max_new_tokens);
        let forward = |model: &MetaIModel, ids: &[u32]| {
            let x: Vec<_> = ids.iter().map(|&i| i as i32).collect();
            model.forward(
                Tensor::<1, Int>::from_ints(x.as_slice(), &device).reshape([1, x.len()]),
                None,
                None,
            )
        };
        while context.len() < end {
            let start = context.len();
            let mut draft = context.clone();
            for _ in 0..lookahead.min(end - start) {
                let len = draft.len();
                let proposal = forward(&self.draft_model, &draft)
                    .slice([0..1, len - 1..len])
                    .argmax(2)
                    .into_scalar::<u32>();
                draft.push(proposal);
            }
            // All proposed tokens are checked in a single causal target pass.
            let input_len = draft.len() - 1;
            let predictions: Vec<_> = forward(&self.target_model, &draft[..input_len])
                .slice([0..1, start - 1..input_len])
                .argmax(2)
                .into_data()
                .iter::<u32>()
                .collect();
            for (candidate, next) in draft[start..].iter().zip(predictions) {
                context.push(next);
                if Some(next) == self.tokenizer.eos_id() || Some(next) == self.tokenizer.pad_id() {
                    return self.tokenizer.decode(&context);
                }
                if *candidate != next {
                    break;
                }
            }
        }
        self.tokenizer.decode(&context)
    }
}
