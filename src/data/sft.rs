use burn::data::dataset::Dataset;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::data::MetaITokenizer;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct InstructionItem {
    pub instruction: String,
    pub input: Option<String>,
    pub output: String,
}

#[derive(Clone)]
pub struct InstructionDataset {
    data: Vec<(Vec<u32>, Vec<u32>)>, // (Token IDs, Loss Masks)
}

impl InstructionDataset {
    pub fn from_file(
        path: &str,
        tokenizer: &MetaITokenizer,
        max_length: usize,
    ) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut data = Vec::new();

        // Chat Template 标记
        // 假设 tokenizer 已经有了这些特殊 token，或者我们硬编码字符串
        // 这里为了通用性，直接使用字符串拼接，依赖 tokenizer 编码
        // 格式: <|user|>\n{instruction}\n<|assistant|>\n{output}<|endoftext|>

        let user_tag = "<|user|>\n";
        let assistant_tag = "\n<|assistant|>\n";
        let eos_tag = "<|endoftext|>";

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            // Parse JSONL
            let item: InstructionItem = serde_json::from_str(&line)
                .map_err(|e| anyhow::anyhow!("Failed to parse JSON line: {}", e))?;

            // 1. 构建 Prompt (Instruction) 部分 -> Mask = 0
            let input_content = item.input.unwrap_or_default();
            let prompt_str = format!(
                "{}{}{}{}",
                user_tag, item.instruction, input_content, assistant_tag
            );
            let prompt_ids = tokenizer.encode(&prompt_str);

            // 2. 构建 Response (Output) 部分 -> Mask = 1
            let response_str = format!("{}{}", item.output, eos_tag);
            let response_ids = tokenizer.encode(&response_str);

            if prompt_ids.is_empty() || response_ids.is_empty() {
                continue;
            }

            // 3. 拼接与截断
            let mut full_ids = prompt_ids.clone();
            full_ids.extend(&response_ids);

            let mut mask = vec![0u32; prompt_ids.len()];
            mask.extend(vec![1u32; response_ids.len()]);

            // Truncate if needed
            if full_ids.len() > max_length {
                // SFT 通常截断尾部，或者只训练部分
                // 这里简单截断，保留前面的 Instruction 和部分 Output
                full_ids.truncate(max_length);
                mask.truncate(max_length);
            }

            // 4. Filter short sequences
            // 至少要有 1 个 token 用于训练 (Output 部分至少包含 1 个)
            if mask.iter().sum::<u32>() > 0 {
                data.push((full_ids, mask));
            }
        }

        Ok(Self { data })
    }
}

impl Dataset<(Vec<u32>, Vec<u32>)> for InstructionDataset {
    fn get(&self, index: usize) -> Option<(Vec<u32>, Vec<u32>)> {
        self.data.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

#[derive(Clone)]
pub struct SFTBatcher<B: Backend> {
    pad_id: u32,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> SFTBatcher<B> {
    pub fn new(_device: B::Device, pad_id: u32) -> Self {
        Self {
            pad_id,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SFTBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub mask: Tensor<B, 2, Int>, // 0 for ignored, 1 for active loss
}

impl<B: Backend> Batcher<B, (Vec<u32>, Vec<u32>), SFTBatch<B>> for SFTBatcher<B> {
    fn batch(&self, items: Vec<(Vec<u32>, Vec<u32>)>, device: &B::Device) -> SFTBatch<B> {
        if items.is_empty() {
            let empty = Tensor::zeros([0, 0], device);
            return SFTBatch {
                inputs: empty.clone(),
                targets: empty.clone(),
                mask: empty,
            };
        }

        let max_original_len = items.iter().map(|(id, _)| id.len()).max().unwrap_or(0);
        let max_len = max_original_len.saturating_sub(1);
        let batch_size = items.len();

        if max_len == 0 {
            let empty = Tensor::zeros([batch_size, 0], device);
            return SFTBatch {
                inputs: empty.clone(),
                targets: empty.clone(),
                mask: empty,
            };
        }

        let mut inputs_data = Vec::with_capacity(batch_size * max_len);
        let mut targets_data = Vec::with_capacity(batch_size * max_len);
        let mut mask_data = Vec::with_capacity(batch_size * max_len);

        for (item, item_mask) in items {
            let item_len = item.len();
            let seq_len = item_len.saturating_sub(1);

            if seq_len == 0 {
                for _ in 0..max_len {
                    inputs_data.push(self.pad_id as i32);
                    targets_data.push(self.pad_id as i32);
                    mask_data.push(0); // Pad mask is 0
                }
                continue;
            }

            // Inputs: [0 .. N-1]
            let mut input: Vec<i32> = item[..seq_len].iter().map(|&x| x as i32).collect();

            // Targets: [1 .. N]
            let mut target: Vec<i32> = item[1..item_len].iter().map(|&x| x as i32).collect();

            // Mask: Aligned with Targets [1 .. N]
            let mut mask: Vec<i32> = item_mask[1..item_len].iter().map(|&x| x as i32).collect();

            // Padding
            while input.len() < max_len {
                input.push(self.pad_id as i32);
            }
            while target.len() < max_len {
                target.push(self.pad_id as i32);
            }
            while mask.len() < max_len {
                mask.push(0); // Pad mask is 0
            }

            inputs_data.extend(input);
            targets_data.extend(target);
            mask_data.extend(mask);
        }

        let inputs = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(inputs_data, [batch_size, max_len]),
            device,
        );

        let targets = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(targets_data, [batch_size, max_len]),
            device,
        );

        let mask = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(mask_data, [batch_size, max_len]),
            device,
        );

        SFTBatch {
            inputs,
            targets,
            mask,
        }
    }
}
