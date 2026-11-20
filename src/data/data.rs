use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, DatasetError};
use burn::tensor::{Int, Tensor};
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::data::MetaITokenizer;

#[derive(Clone)]
pub struct TextDataset {
    data: Vec<Vec<u32>>,
}

impl TextDataset {
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
            let ids = tokenizer.encode(&line);
            for chunk in ids.chunks(max_length) {
                if chunk.len() > 1 {
                    data.push(chunk.to_vec());
                }
            }
        }

        Ok(Self { data })
    }
}

impl Dataset<Vec<u32>> for TextDataset {
    fn get(&self, index: usize) -> Result<Vec<u32>, DatasetError> {
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

/// Disk-backed token blocks. Workers open independent cursors on one immutable token file.
pub struct LazyTextDataset {
    tokens: tempfile::NamedTempFile,
    blocks: Vec<(u64, usize, usize)>, // byte offset, token count, non-padding targets
}

impl LazyTextDataset {
    pub fn new(path: &str, tokenizer: &MetaITokenizer, max_length: usize) -> anyhow::Result<Self> {
        Self::from_files(&[std::path::Path::new(path)], tokenizer, max_length)
    }

    pub fn from_files(
        paths: &[&std::path::Path],
        tokenizer: &MetaITokenizer,
        max_length: usize,
    ) -> anyhow::Result<Self> {
        use std::io::{BufWriter, Write};
        anyhow::ensure!(
            max_length >= 2,
            "token block must have at least two positions"
        );
        let tokens = tempfile::NamedTempFile::new()?;
        let mut writer = BufWriter::new(tokens.as_file());
        let mut blocks = Vec::new();
        let mut offset = 0;
        let pad = tokenizer.pad_id();
        for path in paths {
            for line in BufReader::new(File::open(path)?).lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                for chunk in tokenizer.encode(&line).chunks(max_length) {
                    if chunk.len() < 2 {
                        continue;
                    }
                    let targets = chunk[1..].iter().filter(|&&id| Some(id) != pad).count();
                    if targets == 0 {
                        continue;
                    }
                    blocks.push((offset, chunk.len(), targets));
                    for id in chunk {
                        writer.write_all(&id.to_le_bytes())?;
                    }
                    offset += 4 * chunk.len() as u64;
                }
            }
        }
        writer.flush()?;
        drop(writer);
        Ok(Self { tokens, blocks })
    }

    pub fn targets(&self, index: usize) -> usize {
        self.blocks[index].2
    }

    fn read(&self, file: &mut File, index: usize) -> std::io::Result<Vec<u32>> {
        use std::io::{Read, Seek, SeekFrom};
        let &(offset, length, _) = self
            .blocks
            .get(index)
            .ok_or_else(|| std::io::Error::other("invalid block index"))?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; length * 4];
        file.read_exact(&mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(u32::from_le_bytes)
            .collect())
    }

    pub fn batch_rows(&self, indices: &[usize]) -> std::io::Result<Vec<Vec<u32>>> {
        let mut reader = self.tokens.reopen()?;
        indices.iter().map(|&i| self.read(&mut reader, i)).collect()
    }
}

impl Dataset<Vec<u32>> for LazyTextDataset {
    fn get(&self, index: usize) -> Result<Vec<u32>, DatasetError> {
        self.tokens
            .reopen()
            .and_then(|mut file| self.read(&mut file, index))
            .map_err(DatasetError::new)
    }
    fn len(&self) -> usize {
        self.blocks.len()
    }
}

/// 文本批处理器
///
/// 将变长的 token 序列批处理为固定长度的张量，使用 padding 填充。
/// 同时生成输入和目标序列（目标序列是输入序列向右偏移一位）。
#[derive(Clone)]
pub struct TextBatcher {
    pad_id: u32,
}

impl TextBatcher {
    /// 创建新的批处理器
    ///
    /// # 参数
    /// - `_device`: 设备（当前未使用，保留用于未来扩展）
    /// - `pad_id`: 用于填充的 token ID
    pub fn new(_device: burn::tensor::Device, pad_id: u32) -> Self {
        Self { pad_id }
    }
}

#[derive(Debug, Clone)]
pub struct TextBatch {
    pub inputs: Tensor<2, Int>,
    pub targets: Tensor<2, Int>,
}

impl Batcher<Vec<u32>, TextBatch> for TextBatcher {
    /// 批处理多个 token 序列
    ///
    /// 将变长序列批处理为固定长度，生成输入-目标对用于语言模型训练。
    /// 目标序列是输入序列向右偏移一位（下一个 token 预测任务）。
    fn batch(&self, items: Vec<Vec<u32>>, device: &burn::tensor::Device) -> TextBatch {
        if items.is_empty() {
            // 处理空批次
            let empty = Tensor::<2, Int>::zeros([0, 0], device);
            return TextBatch {
                inputs: empty.clone(),
                targets: empty,
            };
        }

        // 计算最大序列长度（减1，因为输入和目标都比原序列短1）
        let max_original_len = items.iter().map(|item| item.len()).max().unwrap_or(0);
        let max_len = max_original_len.saturating_sub(1);
        let batch_size = items.len();

        if max_len == 0 {
            // 所有序列都太短
            let empty = Tensor::<2, Int>::zeros([batch_size, 0], device);
            return TextBatch {
                inputs: empty.clone(),
                targets: empty,
            };
        }

        // 预分配内存以提高性能
        let mut inputs_data = Vec::with_capacity(batch_size * max_len);
        let mut targets_data = Vec::with_capacity(batch_size * max_len);

        for item in items {
            // 输入：去掉最后一个 token（作为目标）
            // 目标：去掉第一个 token（作为输入）
            // 这样目标[i] = 输入[i] 的下一个 token
            let item_len = item.len();
            let seq_len = item_len.saturating_sub(1);

            if seq_len == 0 {
                // 序列太短，跳过
                for _ in 0..max_len {
                    inputs_data.push(self.pad_id as i32);
                    targets_data.push(self.pad_id as i32);
                }
                continue;
            }

            // 构建输入序列（去掉最后一个）
            let mut input: Vec<i32> = item[..seq_len].iter().map(|&x| x as i32).collect();

            // 构建目标序列（去掉第一个）
            let mut target: Vec<i32> = item[1..item_len].iter().map(|&x| x as i32).collect();

            // 填充到最大长度
            while input.len() < max_len {
                input.push(self.pad_id as i32);
            }
            while target.len() < max_len {
                target.push(self.pad_id as i32);
            }

            inputs_data.extend(input);
            targets_data.extend(target);
        }

        let inputs = Tensor::<2, Int>::from_data(
            burn::tensor::TensorData::new(inputs_data, [batch_size, max_len]),
            device,
        );

        let targets = Tensor::<2, Int>::from_data(
            burn::tensor::TensorData::new(targets_data, [batch_size, max_len]),
            device,
        );

        TextBatch { inputs, targets }
    }
}
