use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::data::MetaITokenizer;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PreferenceItem {
    pub instruction: String,
    pub input: Option<String>,
    pub chosen: String,
    pub rejected: String,
}

#[derive(Clone)]
pub struct PreferenceDataset {
    // (Chosen IDs, Chosen Mask, Rejected IDs, Rejected Mask)
    data: Vec<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)>,
}

impl PreferenceDataset {
    pub fn from_file(
        path: &str,
        tokenizer: &MetaITokenizer,
        max_length: usize,
    ) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut data = Vec::new();

        let user_tag = "<|user|>\n";
        let assistant_tag = "\n<|assistant|>\n";
        let eos_tag = "<|endoftext|>";

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let item: PreferenceItem = serde_json::from_str(&line)
                .map_err(|e| anyhow::anyhow!("Failed to parse JSON line: {}", e))?;

            let input_content = item.input.unwrap_or_default();
            let prompt_str = format!(
                "{}{}{}{}",
                user_tag, item.instruction, input_content, assistant_tag
            );
            let prompt_ids = tokenizer.encode(&prompt_str);

            // Process Chosen
            let chosen_str = format!("{}{}", item.chosen, eos_tag);
            let chosen_resp_ids = tokenizer.encode(&chosen_str);
            let (chosen_ids, chosen_mask) =
                Self::build_sequence(&prompt_ids, &chosen_resp_ids, max_length);

            // Process Rejected
            let rejected_str = format!("{}{}", item.rejected, eos_tag);
            let rejected_resp_ids = tokenizer.encode(&rejected_str);
            let (rejected_ids, rejected_mask) =
                Self::build_sequence(&prompt_ids, &rejected_resp_ids, max_length);

            if chosen_mask.iter().sum::<u32>() > 0 && rejected_mask.iter().sum::<u32>() > 0 {
                data.push((chosen_ids, chosen_mask, rejected_ids, rejected_mask));
            }
        }

        Ok(Self { data })
    }

    fn build_sequence(
        prompt_ids: &[u32],
        resp_ids: &[u32],
        max_length: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut full_ids = prompt_ids.to_vec();
        full_ids.extend(resp_ids);

        let mut mask = vec![0u32; prompt_ids.len()];
        mask.extend(vec![1u32; resp_ids.len()]);

        if full_ids.len() > max_length {
            full_ids.truncate(max_length);
            mask.truncate(max_length);
        }

        (full_ids, mask)
    }
}

impl Dataset<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> for PreferenceDataset {
    fn get(&self, index: usize) -> Option<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> {
        self.data.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Clone)]
pub struct DPOBatcher<B: Backend> {
    pad_id: u32,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> DPOBatcher<B> {
    pub fn new(_device: B::Device, pad_id: u32) -> Self {
        Self {
            pad_id,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DPOBatch<B: Backend> {
    pub chosen_inputs: Tensor<B, 2, Int>,
    pub chosen_targets: Tensor<B, 2, Int>,
    pub chosen_mask: Tensor<B, 2, Int>,

    pub rejected_inputs: Tensor<B, 2, Int>,
    pub rejected_targets: Tensor<B, 2, Int>,
    pub rejected_mask: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>), DPOBatch<B>>
    for DPOBatcher<B>
{
    fn batch(
        &self,
        items: Vec<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)>,
        device: &B::Device,
    ) -> DPOBatch<B> {
        let (chosen_items, rejected_items): (Vec<_>, Vec<_>) = items
            .into_iter()
            .map(|(c_id, c_mask, r_id, r_mask)| ((c_id, c_mask), (r_id, r_mask)))
            .unzip();

        let chosen_batch = self.batch_one_side(chosen_items, device);
        let rejected_batch = self.batch_one_side(rejected_items, device);

        DPOBatch {
            chosen_inputs: chosen_batch.0,
            chosen_targets: chosen_batch.1,
            chosen_mask: chosen_batch.2,
            rejected_inputs: rejected_batch.0,
            rejected_targets: rejected_batch.1,
            rejected_mask: rejected_batch.2,
        }
    }
}

impl<B: Backend> DPOBatcher<B> {
    fn batch_one_side(
        &self,
        items: Vec<(Vec<u32>, Vec<u32>)>,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        if items.is_empty() {
            let empty = Tensor::zeros([0, 0], device);
            return (empty.clone(), empty.clone(), empty);
        }

        let max_len = items
            .iter()
            .map(|(id, _)| id.len())
            .max()
            .unwrap_or(0)
            .saturating_sub(1);
        let batch_size = items.len();

        if max_len == 0 {
            let empty = Tensor::zeros([batch_size, 0], device);
            return (empty.clone(), empty.clone(), empty);
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
                    mask_data.push(0);
                }
                continue;
            }

            let mut input: Vec<i32> = item[..seq_len].iter().map(|&x| x as i32).collect();
            let mut target: Vec<i32> = item[1..item_len].iter().map(|&x| x as i32).collect();
            let mut mask: Vec<i32> = item_mask[1..item_len].iter().map(|&x| x as i32).collect();

            while input.len() < max_len {
                input.push(self.pad_id as i32);
            }
            while target.len() < max_len {
                target.push(self.pad_id as i32);
            }
            while mask.len() < max_len {
                mask.push(0);
            }

            inputs_data.extend(input);
            targets_data.extend(target);
            mask_data.extend(mask);
        }

        let inputs = Tensor::from_data(
            burn::tensor::TensorData::new(inputs_data, [batch_size, max_len]),
            device,
        );
        let targets = Tensor::from_data(
            burn::tensor::TensorData::new(targets_data, [batch_size, max_len]),
            device,
        );
        let mask = Tensor::from_data(
            burn::tensor::TensorData::new(mask_data, [batch_size, max_len]),
            device,
        );

        (inputs, targets, mask)
    }
}
