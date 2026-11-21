//! # KV Cache（分页存储 + 页读取原语）
//!
//! 存储上把 KV 按 `DEFAULT_PAGE_SIZE` 个 token 切成**物理页**（`CacheBlock`），
//! 写入用 `O(page)` 的增量拼接（对比整段拼接的 `O(context)`）。读取侧提供
//! 两种原语：
//!
//! - [`KVCache::get_full`]：整段拼接（naive dense / 参考路径仍需要）；
//! - [`KVCache::gather_spans`]：**页读取** —— 只把「指定 token 区间集合」
//!   覆盖的页切片拼出来，成本 \(O(\text{请求 token})\) 而非 \(O(\text{总长})\)；
//! - [`KVCache::head_rows`]：每页首个 token 的 K（= 稀疏「块代表」行，
//!   供位置打分，数据量为全量 K 的 \(1/\text{page}\)）。
//!
//! 稀疏解码（`sparse_exec.rs` 的页路径）即由后两个原语组成数据流，从而
//! 消除每步 `get_full` 的 \(O(S\_kv)\) 拷贝 —— 见 docs/INDUSTRIAL.zh-CN.md
//! 「页读取」位。三者都从 `blocks` **现算**（不缓存中间状态），因此
//! `truncate` / `clear` 之后自动保持一致。
//!
//! 约束：本实现假设 batch = 1（自回归解码）；页大小固定 `DEFAULT_PAGE_SIZE`。
//! 页读取路径额外要求稀疏块大小与页大小一致（见 `sparse_exec`），否则回退
//! `get_full` 保持语义等价。

use burn::tensor::Tensor;

/// KV 物理页大小（token 数）。与 `SparseAttentionConfig::block_size` 一致时，
/// 稀疏解码自动启用「页读取」路径。
pub const DEFAULT_PAGE_SIZE: usize = 16;

/// Cache Block keeping fixed size KV tensors
#[derive(Clone, Debug)]
pub struct CacheBlock {
    pub k: Tensor<4>, // [Batch, Heads, page_token_count, Dim]
    pub v: Tensor<4>, // [Batch, Heads, page_token_count, Dim]
    pub filled: usize,
}

impl CacheBlock {
    pub fn new(k: Tensor<4>, v: Tensor<4>, filled: usize) -> Self {
        Self { k, v, filled }
    }
}

/// Paged KV Cache implementation avoiding large contiguous allocations
#[derive(Default)]
pub struct KVCache {
    pub blocks: Vec<CacheBlock>,
    pub head_dim: usize,
    padding: Option<Tensor<2, burn::tensor::Bool>>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            head_dim: 0,
            padding: None,
        }
    }

    /// Maintain key padding across cached chunks only when a pad has occurred.
    pub fn append_padding(
        &mut self,
        mask: Option<Tensor<2, burn::tensor::Bool>>,
        batch: usize,
        chunk: usize,
        device: &burn::tensor::Device,
    ) -> Option<Tensor<4, burn::tensor::Bool>> {
        let past = self.seq_len();
        if self.padding.is_none()
            && !mask
                .as_ref()
                .is_some_and(|m| m.clone().any().into_scalar::<bool>())
        {
            return None;
        }
        let zeros = |n| Tensor::<2>::zeros([batch, n], device).bool();
        let next = mask.unwrap_or_else(|| zeros(chunk));
        let full = if past == 0 {
            next
        } else {
            Tensor::cat(
                vec![self.padding.take().unwrap_or_else(|| zeros(past)), next],
                1,
            )
        };
        self.padding = Some(full.clone());
        Some(full.reshape([batch, 1, 1, past + chunk]))
    }

    /// Add new token KV to the cache
    pub fn update(&mut self, k: Tensor<4>, v: Tensor<4>) {
        let [batch, heads, seq, dim] = k.dims();
        self.head_dim = dim;

        // 逐 token 追加到「末页」（O(page) 拼接），页满则开新页；
        // 语义 = 各页只装 `filled` 个有效 token，物理保持分页。

        let mut remaining_seq = seq;
        let mut offset = 0;

        while remaining_seq > 0 {
            let last_block_full = self
                .blocks
                .last()
                .map(|b| b.filled == DEFAULT_PAGE_SIZE)
                .unwrap_or(true);

            if last_block_full {
                // Case: Start new page
                self.blocks.push(CacheBlock::new(
                    k.clone()
                        .slice([0..batch, 0..heads, offset..offset + 1, 0..dim]),
                    v.clone()
                        .slice([0..batch, 0..heads, offset..offset + 1, 0..dim]),
                    1,
                ));
            } else {
                // Append to last page（O(page) copy）
                let last_idx = self.blocks.len() - 1;
                let last = &self.blocks[last_idx];

                let cur_k = k
                    .clone()
                    .slice([0..batch, 0..heads, offset..offset + 1, 0..dim]);
                let cur_v = v
                    .clone()
                    .slice([0..batch, 0..heads, offset..offset + 1, 0..dim]);

                let new_k = Tensor::cat(vec![last.k.clone(), cur_k], 2);
                let new_v = Tensor::cat(vec![last.v.clone(), cur_v], 2);

                self.blocks[last_idx] = CacheBlock::new(new_k, new_v, last.filled + 1);
            }

            remaining_seq -= 1;
            offset += 1;
        }
    }

    /// 物理页大小（token 数）。稀疏解码在 `block_size == page_size` 时启用
    /// 「页读取」路径（见 [`crate::model::sparse_exec`]）。
    pub fn page_size(&self) -> usize {
        DEFAULT_PAGE_SIZE
    }

    /// 整段拼接（naive dense / 非对齐回退路径用）。成本 \(O(\text{总长})\)，
    /// 稀疏解码的页路径请改用 [`KVCache::gather_spans`]。
    pub fn get_full(&self) -> (Tensor<4>, Tensor<4>) {
        let k_list: Vec<_> = self.blocks.iter().map(|b| b.k.clone()).collect();
        let v_list: Vec<_> = self.blocks.iter().map(|b| b.v.clone()).collect();

        if k_list.is_empty() {
            panic!("Cache is empty, cannot get full tensor");
        }

        (Tensor::cat(k_list, 2), Tensor::cat(v_list, 2))
    }

    /// 每页首个 token 的 K 行：`[Batch, Heads, n_pages, Dim]`。
    ///
    /// 稀疏「位置打分」的块代表 = 块首 token 的 K；当稀疏块与物理页对齐时
    /// （`block_size == page_size`），块首即页首 → 此张量直接作打分 reps，
    /// 数据量只有全量 K 的 \(1/\text{page\_size}\)。
    pub fn head_rows(&self) -> Option<Tensor<4>> {
        if self.blocks.is_empty() {
            return None;
        }
        let rows: Vec<Tensor<4>> = self
            .blocks
            .iter()
            .map(|b| {
                let [batch, heads, _len, dim] = b.k.dims();
                b.k.clone().slice([0..batch, 0..heads, 0..1, 0..dim])
            })
            .collect();
        Some(Tensor::cat(rows, 2))
    }

    /// 形状信息 `(kv_heads, head_dim)`，来自首个物理页（空 cache 返回 `(0, 0)`）。
    pub fn kv_shape(&self) -> (usize, usize) {
        self.blocks
            .first()
            .map(|b| {
                let [_, heads, _, dim] = b.k.dims();
                (heads, dim)
            })
            .unwrap_or((0, 0))
    }

    /// **页读取**：按全局 token 区间集合只取出被覆盖的页切片后拼接。
    ///
    /// - `spans`：`(start, end)` 闭开区间，要求**升序且互不相交**（调用方保证，
    ///   否则输出顺序按传入顺序排，token 次序会错乱）；
    /// - `head`：`Some(g)` 只取第 `g` 个 KV 头（GQA 组内 gather 用，省去读全头），
    ///   `None` 返回全部头；
    /// - 裁剪到 cache 当前长度；每个区间按物理页切割成连续 slice（页内切片），
    ///   成本 \(O(\text{请求 token 数})\)，与 cache 总长无关；
    /// - 返回 `[Batch, (Heads), Σlen, Dim]` 的 K/V（行序 = spans 顺序）。
    pub fn gather_spans(
        &self,
        head: Option<usize>,
        spans: &[(usize, usize)],
    ) -> (Tensor<4>, Tensor<4>) {
        assert!(self.seq_len() > 0, "cache 为空，无法页读取");
        let total = self.seq_len();

        // 页起点表（len = pages + 1，末项 = total）→ 每 span 二分定位起始页。
        let mut offsets = Vec::with_capacity(self.blocks.len() + 1);
        let mut acc = 0usize;
        for b in &self.blocks {
            offsets.push(acc);
            acc += b.filled;
        }
        offsets.push(acc);

        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for &(s, e) in spans {
            let s = s.min(total);
            let e = e.min(total);
            if s >= e {
                continue;
            }
            // 包含 s 的页（partition_point: 首个起点 > s 的前一页）
            let mut page = offsets.partition_point(|&o| o <= s).saturating_sub(1);
            while page < self.blocks.len() && offsets[page] < e {
                let b = &self.blocks[page];
                let p_start = offsets[page];
                let p_end = offsets[page + 1];
                let lo = s.max(p_start);
                let hi = e.min(p_end);
                if lo < hi {
                    let [batch, heads, _len, dim] = b.k.dims();
                    let kp;
                    let vp;
                    match head {
                        Some(g) => {
                            kp = b.k.clone().slice([
                                0..batch,
                                g..g + 1,
                                lo - p_start..hi - p_start,
                                0..dim,
                            ]);
                            vp = b.v.clone().slice([
                                0..batch,
                                g..g + 1,
                                lo - p_start..hi - p_start,
                                0..dim,
                            ]);
                        }
                        None => {
                            kp = b.k.clone().slice([
                                0..batch,
                                0..heads,
                                lo - p_start..hi - p_start,
                                0..dim,
                            ]);
                            vp = b.v.clone().slice([
                                0..batch,
                                0..heads,
                                lo - p_start..hi - p_start,
                                0..dim,
                            ]);
                        }
                    }
                    ks.push(kp);
                    vs.push(vp);
                }
                page += 1;
            }
        }
        (Tensor::cat(ks, 2), Tensor::cat(vs, 2))
    }

    pub fn seq_len(&self) -> usize {
        self.blocks.iter().map(|b| b.filled).sum()
    }

    pub fn clear(&mut self) {
        self.blocks.clear();
    }

    /// Truncate cache to a specific length (used in Speculative Decoding rejection)
    pub fn truncate(&mut self, target_len: usize) {
        if let Some(mask) = self.padding.take() {
            let [b, k] = mask.dims();
            if target_len > 0 {
                self.padding = Some(mask.slice([0..b, 0..target_len.min(k)]));
            }
        }
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

pub struct ModelCache {
    pub layers: Vec<KVCache>,
}

impl ModelCache {
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
