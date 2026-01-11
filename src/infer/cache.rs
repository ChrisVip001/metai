use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

const BLOCK_SIZE: usize = 16;

/// Cache Block keeping fixed size KV tensors
#[derive(Clone, Debug)]
pub struct CacheBlock<B: Backend> {
    pub k: Tensor<B, 4>, // [Batch, Heads, BLOCK_SIZE, Dim]
    pub v: Tensor<B, 4>, // [Batch, Heads, BLOCK_SIZE, Dim]
    pub filled: usize,
}

impl<B: Backend> CacheBlock<B> {
    pub fn new(k: Tensor<B, 4>, v: Tensor<B, 4>, filled: usize) -> Self {
        Self { k, v, filled }
    }
}

/// Paged KV Cache implementation avoiding large contiguous allocations
pub struct KVCache<B: Backend> {
    pub blocks: Vec<CacheBlock<B>>,
    pub head_dim: usize,
}

impl<B: Backend> KVCache<B> {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            head_dim: 0,
        }
    }

    /// Add new token KV to the cache
    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) {
        let [batch, heads, seq, dim] = k.dims();
        self.head_dim = dim;

        // Assumes single token update usually [Batch, Heads, 1, Dim]
        // But logic handles variable length 'seq'

        // We simply push to the last block if it has space, or create new one.
        // For 'seq' > available space, we might split.
        // Simplified: Handle 1 token case or append simply.

        let mut remaining_seq = seq;
        let mut offset = 0;

        while remaining_seq > 0 {
            let last_block_full = self
                .blocks
                .last()
                .map(|b| b.filled == BLOCK_SIZE)
                .unwrap_or(true);

            if last_block_full {
                // Allocate new block (initially empty or partial)
                // Since Burn tensors are immutable, we rely on lazy eval or we just create a "Block"
                // containing the data we have.
                // ideally we want to allocate [BLOCK_SIZE] and fill it.
                // But Burn doesn't support in-place mutation easily.
                // So we store "Partial Blocks" and "Full Blocks"?
                // Or we just store the content and concatenate logically?
                // To achieve "Paged" memory benefit, we should have fixed size tensors.
                // But populating them one by one requires "slice_assign" which copies.

                // Workaround for Burn high-level:
                // Just store chunks as they come? No, that's just List<Tensor>.
                // We want to Group them into chunks of BLOCK_SIZE.

                // If input is 1 token.
                // New Block: create [BLOCK_SIZE] zeros, assign [0] = token.
                // This is expensive to do every step if we copy [BLOCK_SIZE].

                // Optimal High-Level Rust Strategy:
                // `blocks` contains valid tensors of size `filled`.
                // When `filled` reaches `BLOCK_SIZE`, we "Finalize" it (maybe optimize layout).
                // Actually, just storing `Tensor` of valid size is fine, as long as we don't `cat` them all.
                // We merge small chunks into a Block strictly when they fill up?
                // Complex.

                // Let's implement simple Append for now, but enabling Block-wise attention iter.
                // We append `k, v` to a buffer. When buffer == BLOCK_SIZE, push to blocks.
                // WAIT. If we buffer, we lose the info for current step attention!
                // We need the data immediately.

                // So we store `Vec<Tensor>` where most are `BLOCK_SIZE`, last is `< BLOCK_SIZE`.
                // When last becomes `BLOCK_SIZE`, we start a new one.
                // Is `cat(small, token)` better than `cat(huge, token)`? YES. O(BlockSize) vs O(Context).
                // This achieves the PagedAttention speedup for specific kernels!

                // Case: Start new block
                self.blocks.push(CacheBlock::new(
                    k.clone()
                        .slice([0..batch, 0..heads, offset..offset + 1, 0..dim]),
                    v.clone()
                        .slice([0..batch, 0..heads, offset..offset + 1, 0..dim]),
                    1,
                ));
            } else {
                // Append to last block
                let last_idx = self.blocks.len() - 1;
                let last = &self.blocks[last_idx];

                let cur_k = k
                    .clone()
                    .slice([0..batch, 0..heads, offset..offset + 1, 0..dim]);
                let cur_v = v
                    .clone()
                    .slice([0..batch, 0..heads, offset..offset + 1, 0..dim]);

                // O(BlockSize) copy
                let new_k = Tensor::cat(vec![last.k.clone(), cur_k], 2);
                let new_v = Tensor::cat(vec![last.v.clone(), cur_v], 2);

                self.blocks[last_idx] = CacheBlock::new(new_k, new_v, last.filled + 1);
            }

            remaining_seq -= 1;
            offset += 1;
        }
    }

    /// Helper to get full tensor (for compatibility or naive attn)
    /// This is slow O(N) copy, use iter_blocks for fast attn.
    pub fn get_full(&self) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let k_list: Vec<_> = self.blocks.iter().map(|b| b.k.clone()).collect();
        let v_list: Vec<_> = self.blocks.iter().map(|b| b.v.clone()).collect();

        if k_list.is_empty() {
            panic!("Cache is empty, cannot get full tensor");
            // Actually should return None or handle logic outside.
            // Logic in attention handles 'Option<Cache>'.
        }

        (Tensor::cat(k_list, 2), Tensor::cat(v_list, 2))
    }

    pub fn seq_len(&self) -> usize {
        self.blocks.iter().map(|b| b.filled).sum()
    }

    pub fn clear(&mut self) {
        self.blocks.clear();
    }

    /// Truncate cache to a specific length (used in Speculative Decoding rejection)
    pub fn truncate(&mut self, target_len: usize) {
        let mut current_pos = 0;
        let mut blocks_to_keep = 0;
        let mut last_block_new_size = 0;

        for block in &self.blocks {
            if current_pos + block.filled <= target_len {
                current_pos += block.filled;
                blocks_to_keep += 1;
            } else {
                last_block_new_size = target_len - current_pos;
                break;
            }
        }

        // Remove fully invalid blocks
        self.blocks.truncate(if last_block_new_size > 0 {
            blocks_to_keep + 1
        } else {
            blocks_to_keep
        });

        // Slice the last block if it was partially truncated
        if last_block_new_size > 0 {
            if let Some(last) = self.blocks.last_mut() {
                let [batch, heads, _old_len, dim] = last.k.dims();
                last.k = last
                    .k
                    .clone()
                    .slice([0..batch, 0..heads, 0..last_block_new_size, 0..dim]);
                last.v = last
                    .v
                    .clone()
                    .slice([0..batch, 0..heads, 0..last_block_new_size, 0..dim]);
                last.filled = last_block_new_size;
            }
        }
    }
}

pub struct ModelCache<B: Backend> {
    pub layers: Vec<KVCache<B>>,
}

impl<B: Backend> ModelCache<B> {
    pub fn new(num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(KVCache::new());
        }
        Self { layers }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn truncate(&mut self, target_len: usize) {
        for layer in &mut self.layers {
            layer.truncate(target_len);
        }
    }

    pub fn total_seq_len(&self) -> usize {
        // In Transformer, all layers usually have same current seq_len
        self.layers.first().map(|l| l.seq_len()).unwrap_or(0)
    }

    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len()).unwrap_or(0)
    }
}
