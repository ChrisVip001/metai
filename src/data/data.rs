use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::tensor::backend::Backend;
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
    fn get(&self, index: usize) -> Option<Vec<u32>> {
        self.data.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

/// 一个按需从磁盘读取的惰性数据集。
/// 内存占用极小，允许处理大于内存的数据集。
pub struct LazyTextDataset {
    file_path: String,
    indices: Vec<(u64, usize)>, // (偏移量, 字节长度)
    tokenizer: MetaITokenizer,
    max_length: usize,
}

impl LazyTextDataset {
    pub fn new(path: &str, tokenizer: &MetaITokenizer, max_length: usize) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut indices = Vec::new();
        let mut offset = 0;
        let mut line = String::new();

        // 构建索引
        // 注意：这种简单的索引假设 1 行 = 1 个样本。
        // 如果我们想要每行有多个块，我们需要更复杂的索引（line_idx, chunk_idx）。
        // 为了处理大数据时的简单性和效率，我们将每一行视为一个文档。
        // 如果一行太长，我们可能只取第一个块，或者我们需要逻辑映射。
        // 让我们先实现严格的基于行的索引。

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                break;
            }
            if !line.trim().is_empty() {
                indices.push((offset, bytes_read));
            }
            offset += bytes_read as u64;
        }

        Ok(Self {
            file_path: path.to_string(),
            indices,
            tokenizer: tokenizer.clone(),
            max_length,
        })
    }
}

impl Dataset<Vec<u32>> for LazyTextDataset {
    fn get(&self, index: usize) -> Option<Vec<u32>> {
        let (offset, length) = *self.indices.get(index)?;

        // 按需打开文件（操作系统通常会缓存文件句柄/页面）
        // 为了更好的性能，我们可以保留一个线程本地的文件句柄，但与分词相比，文件打开相对便宜。
        let mut file = File::open(&self.file_path).ok()?;
        use std::io::{Read, Seek, SeekFrom};
        file.seek(SeekFrom::Start(offset)).ok()?;

        let mut buffer = vec![0u8; length];
        file.read_exact(&mut buffer).ok()?;

        let text = String::from_utf8(buffer).ok()?;
        let ids = self.tokenizer.encode(&text);

        // 在惰性模式下，为了简单起见，截断到 max_length
        // 或者实现分块逻辑（需要索引扩展 1 行 -> N 个样本）
        // 当前：取前 max_length 个
        let chunk = if ids.len() > self.max_length {
            ids[..self.max_length].to_vec()
        } else {
            ids
        };

        if chunk.len() > 1 {
            Some(chunk)
        } else {
            None // 理论上应该在构建期间通过过滤索引来处理这种情况
        }
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

/// 文本批处理器
/// 
/// 将变长的 token 序列批处理为固定长度的张量，使用 padding 填充。
/// 同时生成输入和目标序列（目标序列是输入序列向右偏移一位）。
#[derive(Clone)]
pub struct TextBatcher<B: Backend> {
    pad_id: u32,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> TextBatcher<B> {
    /// 创建新的批处理器
    /// 
    /// # 参数
    /// - `_device`: 设备（当前未使用，保留用于未来扩展）
    /// - `pad_id`: 用于填充的 token ID
    pub fn new(_device: B::Device, pad_id: u32) -> Self {
        Self {
            pad_id,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, Vec<u32>, TextBatch<B>> for TextBatcher<B> {
    /// 批处理多个 token 序列
    /// 
    /// 将变长序列批处理为固定长度，生成输入-目标对用于语言模型训练。
    /// 目标序列是输入序列向右偏移一位（下一个 token 预测任务）。
    fn batch(&self, items: Vec<Vec<u32>>, device: &B::Device) -> TextBatch<B> {
        if items.is_empty() {
            // 处理空批次
            let empty = Tensor::<B, 2, Int>::zeros([0, 0], device);
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
            let empty = Tensor::<B, 2, Int>::zeros([batch_size, 0], device);
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
            let mut input: Vec<i32> = item[..seq_len]
                .iter()
                .map(|&x| x as i32)
                .collect();
            
            // 构建目标序列（去掉第一个）
            let mut target: Vec<i32> = item[1..item_len]
                .iter()
                .map(|&x| x as i32)
                .collect();

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

        let inputs = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(inputs_data, [batch_size, max_len]),
            device,
        );

        let targets = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(targets_data, [batch_size, max_len]),
            device,
        );

        TextBatch { inputs, targets }
    }
}
