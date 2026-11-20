use burn::data::dataset::{Dataset, DatasetError};
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
    pub fn supervised_targets(&self) -> usize {
        self.data
            .iter()
            .map(|(_, mask)| mask.iter().skip(1).filter(|&&v| v == 1).count())
            .sum()
    }

    pub fn from_file(
        path: &str,
        tokenizer: &MetaITokenizer,
        max_length: usize,
    ) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut data = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            // Parse JSONL
            let item: InstructionItem = serde_json::from_str(&line)
                .map_err(|e| anyhow::anyhow!("Failed to parse JSON line: {}", e))?;

            if let Some(row) = encode_instruction(&item, tokenizer, max_length) {
                data.push(row);
            }
        }

        Ok(Self { data })
    }
}

/// Encode one SFT record and mark only its answer and EOS for next-token loss.
pub fn encode_instruction(
    item: &InstructionItem,
    tokenizer: &MetaITokenizer,
    max_length: usize,
) -> Option<(Vec<u32>, Vec<u32>)> {
    let prompt = super::chat_prompt(&item.instruction, item.input.as_deref().unwrap_or_default());
    let mut ids = tokenizer.encode(&prompt);
    let prompt_len = ids.len();
    if prompt_len == 0 || prompt_len >= max_length {
        return None;
    }
    ids.extend(
        tokenizer
            .response(&item.output)
            .into_iter()
            .take(max_length - prompt_len),
    );
    if ids.len() == prompt_len {
        return None;
    }
    let mut mask = vec![0; prompt_len];
    mask.resize(ids.len(), 1);
    Some((ids, mask))
}

impl Dataset<(Vec<u32>, Vec<u32>)> for InstructionDataset {
    fn get(&self, index: usize) -> Result<(Vec<u32>, Vec<u32>), DatasetError> {
        self.data
            .as_slice()
            .get(index)
            .cloned()
            .ok_or_else(|| DatasetError::new(std::io::Error::other("invalid dataset index")))
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{Int, Tensor};

#[derive(Clone)]
pub struct SFTBatcher {
    pad_id: u32,
}

impl SFTBatcher {
    pub fn new(_device: burn::tensor::Device, pad_id: u32) -> Self {
        Self { pad_id }
    }
}

#[derive(Debug, Clone)]
pub struct SFTBatch {
    pub inputs: Tensor<2, Int>,
    pub targets: Tensor<2, Int>,
    pub mask: Tensor<2, Int>, // 0 for ignored, 1 for active loss
}

impl Batcher<(Vec<u32>, Vec<u32>), SFTBatch> for SFTBatcher {
    fn batch(&self, items: Vec<(Vec<u32>, Vec<u32>)>, device: &burn::tensor::Device) -> SFTBatch {
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

        let inputs = Tensor::<2, Int>::from_data(
            burn::tensor::TensorData::new(inputs_data, [batch_size, max_len]),
            device,
        );

        let targets = Tensor::<2, Int>::from_data(
            burn::tensor::TensorData::new(targets_data, [batch_size, max_len]),
            device,
        );

        let mask = Tensor::<2, Int>::from_data(
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

// ============================ GRPO 数据集 ============================
// GRPO 需要保留 prompt/response 边界（真 rollout 时按 prompt 采样多条 response），
// SFT 的 (ids, mask) 不保留边界，因此单独定义。

#[derive(Clone, Debug)]
pub struct GRPOItem {
    pub answer: String,
    pub instruction: String,
    pub prompt: Vec<u32>,
}

#[derive(Clone)]
pub struct GRPODataset {
    pub data: Vec<GRPOItem>,
}

impl GRPODataset {
    pub fn from_file(
        path: &str,
        tokenizer: &MetaITokenizer,
        max_length: usize,
    ) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut data = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let item: InstructionItem = serde_json::from_str(&line)
                .map_err(|e| anyhow::anyhow!("Failed to parse JSON line: {}", e))?;

            let input_content = item.input.clone().unwrap_or_default();
            let prompt_str = super::chat_prompt(&item.instruction, &input_content);
            let prompt_ids = tokenizer.encode(&prompt_str);

            anyhow::ensure!(!item.output.trim().is_empty(), "empty reward target");
            anyhow::ensure!(
                !prompt_ids.is_empty() && prompt_ids.len() < max_length,
                "RL prompt exhausts context"
            );
            let question = if input_content.is_empty() {
                item.instruction
            } else {
                format!("{}\n{}", item.instruction, input_content)
            };
            data.push(GRPOItem {
                answer: item.output,
                instruction: question,
                prompt: prompt_ids,
            });
        }

        Ok(Self { data })
    }
}

impl Dataset<GRPOItem> for GRPODataset {
    fn get(&self, index: usize) -> Result<GRPOItem, DatasetError> {
        self.data
            .as_slice()
            .get(index)
            .cloned()
            .ok_or_else(|| DatasetError::new(std::io::Error::other("invalid dataset index")))
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}
