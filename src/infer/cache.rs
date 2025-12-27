use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// KV Cache 用于存储注意力层的 Key 和 Value
/// 
/// 在增量推理中，只需要计算新 token 的 KV，然后与历史 KV 拼接。
/// 这样可以避免重复计算，大幅提升推理速度。
pub struct KVCache<B: Backend> {
    /// Key 缓存，形状为 [Batch, Heads, Seq, Dim]
    pub k: Option<Tensor<B, 4>>,
    /// Value 缓存，形状为 [Batch, Heads, Seq, Dim]
    pub v: Option<Tensor<B, 4>>,
}

impl<B: Backend> KVCache<B> {
    /// 创建新的空 KV Cache
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// 更新 KV Cache，将新的 KV 与历史 KV 拼接
    /// 
    /// # 参数
    /// - `k`: 新的 Key 张量，形状为 [Batch, Heads, NewSeq, Dim]
    /// - `v`: 新的 Value 张量，形状为 [Batch, Heads, NewSeq, Dim]
    /// 
    /// # 返回
    /// 拼接后的完整 KV，形状为 [Batch, Heads, TotalSeq, Dim]
    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (new_k, new_v) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                // 拼接历史 KV，沿序列维度（dim=2）拼接
                let k_concat = Tensor::cat(vec![prev_k.clone(), k], 2);
                let v_concat = Tensor::cat(vec![prev_v.clone(), v], 2);
                (k_concat, v_concat)
            }
            _ => (k, v),
        };

        // 更新缓存（只保留引用，避免不必要的克隆）
        self.k = Some(new_k.clone());
        self.v = Some(new_v.clone());

        (new_k, new_v)
    }

    /// 清空缓存，释放内存
    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// 获取当前缓存的序列长度
    pub fn seq_len(&self) -> usize {
        self.k.as_ref()
            .map(|k| k.dims()[2])
            .unwrap_or(0)
    }
}

/// 模型级别的 KV Cache，包含所有层的缓存
/// 
/// 每个 Transformer 层都有独立的 KV Cache，用于存储该层的注意力状态。
pub struct ModelCache<B: Backend> {
    /// 每一层的 KV Cache
    pub layers: Vec<KVCache<B>>,
}

impl<B: Backend> ModelCache<B> {
    /// 创建新的模型级 Cache
    /// 
    /// # 参数
    /// - `num_layers`: Transformer 层数
    pub fn new(num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(KVCache::new());
        }
        Self { layers }
    }

    /// 清空所有层的缓存
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// 获取总缓存大小（所有层的序列长度之和）
    pub fn total_seq_len(&self) -> usize {
        self.layers.iter().map(|l| l.seq_len()).sum()
    }
}
