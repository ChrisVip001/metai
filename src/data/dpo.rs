use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, DatasetError};
use burn::tensor::{Int, Tensor};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::data::{sft::SFTBatcher, MetaITokenizer};

type PreferencePair = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);

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
    data: Vec<PreferencePair>,
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

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let item: PreferenceItem = serde_json::from_str(&line)
                .map_err(|e| anyhow::anyhow!("Failed to parse JSON line: {}", e))?;

            let input_content = item.input.unwrap_or_default();
            let prompt_str = super::chat_prompt(&item.instruction, &input_content);
            let prompt_ids = tokenizer.encode(&prompt_str);

            // Process Chosen
            let chosen_resp_ids = tokenizer.response(&item.chosen);
            let (chosen_ids, chosen_mask) =
                Self::build_sequence(&prompt_ids, &chosen_resp_ids, max_length);

            // Process Rejected
            let rejected_resp_ids = tokenizer.response(&item.rejected);
            let (rejected_ids, rejected_mask) =
                Self::build_sequence(&prompt_ids, &rejected_resp_ids, max_length);

            if chosen_ids != rejected_ids
                && chosen_mask.iter().sum::<u32>() > 0
                && rejected_mask.iter().sum::<u32>() > 0
            {
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

impl Dataset<PreferencePair> for PreferenceDataset {
    fn get(&self, index: usize) -> Result<PreferencePair, DatasetError> {
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

#[derive(Clone)]
pub struct DPOBatcher {
    pad_id: u32,
}

impl DPOBatcher {
    pub fn new(_device: burn::tensor::Device, pad_id: u32) -> Self {
        Self { pad_id }
    }
}

#[derive(Debug, Clone)]
pub struct DPOBatch {
    pub chosen_inputs: Tensor<2, Int>,
    pub chosen_targets: Tensor<2, Int>,
    pub chosen_mask: Tensor<2, Int>,

    pub rejected_inputs: Tensor<2, Int>,
    pub rejected_targets: Tensor<2, Int>,
    pub rejected_mask: Tensor<2, Int>,
}

impl Batcher<PreferencePair, DPOBatch> for DPOBatcher {
    fn batch(&self, items: Vec<PreferencePair>, device: &burn::tensor::Device) -> DPOBatch {
        let (chosen_items, rejected_items): (Vec<_>, Vec<_>) = items
            .into_iter()
            .map(|(c_id, c_mask, r_id, r_mask)| ((c_id, c_mask), (r_id, r_mask)))
            .unzip();

        let batcher = SFTBatcher::new(device.clone(), self.pad_id);
        let chosen = batcher.batch(chosen_items, device);
        let rejected = batcher.batch(rejected_items, device);
        DPOBatch {
            chosen_inputs: chosen.inputs,
            chosen_targets: chosen.targets,
            chosen_mask: chosen.mask,
            rejected_inputs: rejected.inputs,
            rejected_targets: rejected.targets,
            rejected_mask: rejected.mask,
        }
    }
}
