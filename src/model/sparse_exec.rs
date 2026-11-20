//! # SparseExec —— 稀疏注意力的「物理执行位」（紧凑 gather）
//!
//! [`crate::model::sparse`] 把「位置打分 + Top-k + 窗口」落成了 **mask 语义**
//! （`[B,H,Sq,S_kv]`，随后喂 naive dense SDPA —— 计算仍是 \(O(S^2)\)）。
//! 本文件是它的**执行侧**：在推理解码（自回归，每步一个 token）时**物理上不
//! 计算被屏蔽位置的分数**，只对被选中的块 ∪ 窗口内的 KV 做一次小规模 SDPA。
//!
//! ## 流水线（对应 sparse.rs 的三段算子碎片）
//!
//! ```text
//! ① 位置打分   q 与「每块代表 K」打分（reps 每块只有 1 行 → O(N=S_kv/bs)）
//! ② Top-k      GPU 选出每 query head 分最高的 k 块 → 索引回 CPU 组装
//! ③ 紧凑 gather  每 KV 头组：组内 query head 的 top-k 块 ∪ 滑动窗口
//!                → 展开成 token 列集合 → 页读取 K/V → [1,1,L,d]
//! ④ 紧凑 SDPA    组内 repeat 后对 L ≪ S_kv 的矩阵做 dense SDPA
//! ```
//!
//! 其中 \(L \le (n\_rep\cdot k)\cdot bs + \text{window}\)（组内并集，未去重的
//! 最坏上界），相比 dense 解码的 \(O(S\_kv)\) 打分 / softmax / 加权，物理计算
//! 量降到 \(O(L)\)——即 ARCHITECTURE_TIMELINE §9 的 **kernel 位（gather 版）**。
//!
//! ## 数据源：全量拼接 vs 页读取（两个入口）
//!
//! - [`sparse_decode_attention`]：K/V 以 `[1,Hkv,S_kv,D]` 全量张量传入
//!   （由上层 `get_full()` 拼好）——测试与非对齐回退用；
//! - [`sparse_decode_attention_cached`]：直接以 [`crate::infer::cache::KVCache`]
//!   为数据源。当稀疏块与 cache 物理页对齐（`block_size == page_size`）时走
//!   **页路径**：位置打分的块代表 = 每页首行（`KVCache::head_rows`，数据量为
//!   全量 K 的 \(1/\text{page}\)），组 gather 只读选中页（`KVCache::gather_spans`，
//!   \(O(L)\) 而非 \(O(S\_kv)\)）——**消除了每步 `get_full` 的全量拼接**；
//!   不对齐时自动回退全量路径，语义保持一致。
//!
//! ## 数学等价性
//!
//! 对可见集 \(\mathcal V\)（= 该 query head 的 top-k 块 token ∪ 窗口），
//! softmax 只在这些位置上归一化、其余为 `-inf`。因此与
//! `sparse_attention_mask` + dense SDPA **逐位一致**（本文件一致性测试锁定），
//! 也与 prefill 的 mask 语义一致（由 `attention.rs` 的 prefill/decode 对照测试
//! 保证）。mask 位是「参考实现」，本文件是「执行实现」，两者必须等价。
//!
//! ## 已知边界（诚实声明）
//!
//! - 仅覆盖 **decode 单行**（`Batch == 1 && Sq == 1`，自回归主路径）。
//!   prefill / 多行验证的变长可见集仍走 mask 位（chunked-prefill 为后续位）；
//! - 索引组装在 CPU 进行（每层每步一次 GPU→CPU 小同步：`H×k` 个块号）：
//!   - 短处：层数 × 步数的同步开销（`Generator` 每步本就要同步一次 logits，
//!     故同数量级）；真正零同步需单 kernel 融合（cubecl，后续位）；
//!   - 长处：索引选择逻辑（去重/排序/窗口）可读、可测、与后端无关；
//! - 页路径要求 `cfg.block_size == cache.page_size()`（默认 16）；其它块粒度
//!   自动回退 `get_full`（语义正确，仅不享受页收益）；
//! - 页路径的位置打分仍需遍历全部页首行（`O(S_kv/16)` 的数学量 + 一次小 cat），
//!   这是「块级检索」本身不可省的打分成本（工业实现以 paged kernel 融合）；
//! - 建议 `window >= 1`：`window == 0` 且当前 token 所在块未被 Top-k 选中时，
//!   该位置不可见（与 mask 位语义一致，训练如此则无需特殊处理；见 sparse.rs）。
//!
//! ## 形状约束
//!
//! - `q`：`[1, Heads, 1, HeadDim]`（已 RoPE，**query head 布局，未做 GQA 扩展**）；
//! - 全量入口 k/v：`[1, Heads_kv, S_kv, HeadDim]`（KV 头布局）；
//! - `n_rep = Heads / Heads_kv`（GQA 组大小，本文件按 KV 头分组处理）；
//! - 返回：`[1, Heads, 1, HeadDim]`（可直接接 `merge_heads` + 输出投影）。

use burn::tensor::{Bool, Int, Tensor, TensorData};

use crate::infer::cache::KVCache;
use crate::model::heads::repeat_kv;
use crate::model::sdpa::scaled_dot_product_attention;
use crate::model::sparse::SparseAttentionConfig;

/// 解码单行的稀疏注意力（全量拼接数据源版，供测试与回退）。
///
/// `k / v` 为已含当前 token 的全量 KV（`[1, Heads_kv, S_kv, D]`）。
/// 语义等价于参考实现（mask 位，offset = S_kv - 1）。
pub fn sparse_decode_attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    n_rep: usize,
    cfg: &SparseAttentionConfig,
) -> Tensor<4> {
    let [batch, heads_kv, _kv_len, head_dim] = k.dims();
    assert_eq!(batch, 1, "紧凑稀疏解码只支持 batch = 1");
    assert_eq!(heads_kv * n_rep, q.dims()[1], "GQA 头数不匹配");
    let _ = head_dim;
    decode_with_full_kv(q, k, v, heads_kv, n_rep, cfg)
}

/// 解码单行的稀疏注意力（cache 数据源版，自动选择页 / 回退路径）。
///
/// - `cfg.block_size == cache.page_size()`（默认 16）→ **页路径**：位置打分用
///   `head_rows`（页首行），组 gather 用 `gather_spans`（只读选中页），
///   消除每步 `get_full` 的 `O(S_kv)` 全量拼接；
/// - 否则 → 回退 `cache.get_full()` + 全量入口（语义不变，仅不享页收益）。
pub fn sparse_decode_attention_cached(
    q: Tensor<4>,
    cache: &KVCache,
    n_rep: usize,
    cfg: &SparseAttentionConfig,
) -> Tensor<4> {
    let [batch, heads_q, sq, head_dim] = q.dims();
    assert_eq!(batch, 1, "紧凑稀疏解码只支持 batch = 1");
    assert_eq!(sq, 1, "紧凑稀疏解码只支持单行 query（每步一个 token）");
    let (heads_kv, cache_dim) = cache.kv_shape();
    assert_eq!(heads_kv * n_rep, heads_q, "GQA 头数不匹配");
    assert_eq!(cache_dim, head_dim, "cache 维度与 query 不一致");
    let kv_len = cache.seq_len();
    assert!(kv_len > 0, "cache 为空不能做稀疏解码");

    if cfg.block_size.max(1) == cache.page_size() {
        decode_with_pages(q, cache, heads_kv, n_rep, cfg, kv_len)
    } else {
        let (k, v) = cache.get_full();
        decode_with_full_kv(q, k, v, heads_kv, n_rep, cfg)
    }
}

// ---------------------------------------------------------------------------
// 内部实现
// ---------------------------------------------------------------------------

/// 从「得分 reps」选出每 query 头的 top-k 块号（回 CPU）。
/// `reps`：[1, Hkv, N, D]（每块代表行，KV 头布局）；内部重复到 query 头后打分
/// `[1, Hq, 1, N]`，Top-k 取每 query 头分最高的 N 中 k 个。打分未缩放
/// （缩放因子不影响 Top-k 次序）。
fn topk_block_ids(q: &Tensor<4>, reps: Tensor<4>, n_rep: usize, top_k: usize) -> Vec<Vec<usize>> {
    let [_, heads_q, _, _] = q.dims();
    let reps_q = repeat_kv(reps, n_rep); // [1, Hq, N, D]
    let scores = q.clone().matmul(reps_q.swap_dims(2, 3)); // [1, Hq, 1, N]
    let (_, topk_indices) = scores.topk_with_indices(top_k, 3); // [1, Hq, 1, k]
                                                                // Int 元素在后端间不同（wgpu = i32，torch = i64）→ 尝试后兜底转换。
    let topk_data = topk_indices.into_data();
    let topk_flat: Vec<i64> = match topk_data.try_to_vec::<i64>() {
        Ok(v) => v,
        Err(_) => topk_data
            .try_to_vec::<i32>()
            .map(|v| v.into_iter().map(|x| x as i64).collect())
            .unwrap(),
    };
    // 展平顺序（row-major，[1,Hq,1,k]）：h × k + t
    (0..heads_q)
        .map(|h| {
            (0..top_k)
                .map(|t| topk_flat[h * top_k + t] as usize)
                .collect()
        })
        .collect()
}

/// 页路径：块 == 物理页（`cfg.block_size == page_size` 已由入口保证）。
///
/// 差异点（对比全量路径）：① 打分 reps 直接取每页首行（无补齐/slice）；
/// ③ 组 gather 把 token 列集转成连续区间，走 `cache.gather_spans`（页读取，
/// 成本 O(L)）；其余（Top-k 语义、窗口、per-head mask、SDPA）与全量路径一致。
fn decode_with_pages(
    q: Tensor<4>,
    cache: &KVCache,
    heads_kv: usize,
    n_rep: usize,
    cfg: &SparseAttentionConfig,
    kv_len: usize,
) -> Tensor<4> {
    let [batch, _heads_q, sq, head_dim] = q.dims();
    let device = q.device();
    let scale = (head_dim as f32).sqrt().recip();
    let page = cache.page_size();
    let n_pages = cache.head_rows().map(|r| r.dims()[2]).unwrap_or(0);
    let top_k = cfg.top_k_blocks.max(1).min(n_pages.max(1));

    // ①+② 打分：块代表 = 每页首行（数据量为全量 K 的 1/page）
    let reps = cache
        .head_rows()
        .expect("decode 时 cache 非空，页首行必须存在");
    let topk_per_head = topk_block_ids(&q, reps, n_rep, top_k);

    // 当前 token 全局位置 = cache 更新后的最后一行（解码主路径约定）。
    let pos = kv_len - 1;
    let win_start = pos.saturating_sub(cfg.window);
    // 页 b 覆盖的 token 列 = [b·page, min((b+1)·page, kv_len))（非末页必满）。
    let page_cols = |b: usize| b * page..((b + 1) * page).min(kv_len);

    let mut out_parts = Vec::with_capacity(heads_kv);
    for g in 0..heads_kv {
        let h0 = g * n_rep;
        let h1 = h0 + n_rep;

        // ③a 组内并集列（升序去重）：组内 query head 的 top-k 页 token ∪ 窗口
        let mut cols: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for selected in &topk_per_head[h0..h1] {
            for &b in selected {
                cols.extend(page_cols(b));
            }
        }
        cols.extend(win_start..=pos);
        let cols: Vec<usize> = cols.into_iter().collect();
        let l = cols.len();
        assert!(l > 0, "KV 头组 {g} 可见列为空（kv_len={kv_len}）");

        // ③b 列集合 → 连续区间（升序、互不相交）→ 页读取该组所需 KV
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(4);
        for &c in &cols {
            match spans.last_mut() {
                Some((_, e)) if *e == c => *e = c + 1,
                _ => spans.push((c, c + 1)),
            }
        }
        let (k_g0, v_g0) = cache.gather_spans(Some(g), &spans); // [1,1,L,d]
        let k_g = repeat_kv(k_g0, n_rep); // [1,n,L,d]
        let v_g = repeat_kv(v_g0, n_rep);

        // ④ per-head 可见性 mask：列对本 head 有效 ⇔ 属于其 top-k 页或窗口内。
        // 组内并集列中「别家 head 选的列」对本 head 无效 → -inf 屏蔽。
        let mut mask_flat = Vec::with_capacity(n_rep * l);
        for selected in &topk_per_head[h0..h1] {
            let in_block = |c: usize| selected.iter().any(|&b| page_cols(b).contains(&c));
            for &c in &cols {
                let visible = c >= win_start || in_block(c);
                mask_flat.push(!visible); // SDPA 语义：true = 屏蔽
            }
        }
        let mask_t: Tensor<4, Bool> =
            Tensor::from_data(TensorData::new(mask_flat, [1, n_rep, 1, l]), &device);

        let q_g = q.clone().slice([0..batch, h0..h1, 0..sq, 0..head_dim]);
        let out_g = scaled_dot_product_attention(q_g, k_g, v_g, scale, Some(mask_t));
        out_parts.push(out_g);
    }
    Tensor::cat(out_parts, 1) // [1, Hq, 1, d]
}

/// 全量路径：K/V 以 `[1, Hkv, S_kv, D]` 传入。
///
/// 差异点（对比页路径）：① 打分 reps 从全量 K 补齐末块后取每块首行；
/// ③ 组 gather 用 `select`（token 列索引）抽取单 KV 头的行。
fn decode_with_full_kv(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    heads_kv: usize,
    n_rep: usize,
    cfg: &SparseAttentionConfig,
) -> Tensor<4> {
    let [batch, _heads_q, sq, head_dim] = q.dims();
    let [_, _, kv_len, _] = k.dims();
    let device = q.device();
    let scale = (head_dim as f32).sqrt().recip();
    let bs = cfg.block_size.max(1);
    let n_blocks = kv_len.div_ceil(bs);
    let top_k = cfg.top_k_blocks.max(1).min(n_blocks.max(1));

    // ①+② 打分：块代表 = 每逻辑块首 token 的 K（补齐末块后取第 0 行）
    let padded_len = n_blocks * bs;
    let k_padded = if padded_len > kv_len {
        let zeros = Tensor::<4>::zeros([batch, heads_kv, padded_len - kv_len, head_dim], &device);
        Tensor::cat(vec![k.clone(), zeros], 2)
    } else {
        k.clone()
    };
    let reps = k_padded
        .reshape([batch, heads_kv, n_blocks, bs, head_dim])
        .slice([0..batch, 0..heads_kv, 0..n_blocks, 0..1, 0..head_dim])
        .reshape([batch, heads_kv, n_blocks, head_dim]);
    let topk_per_head = topk_block_ids(&q, reps, n_rep, top_k);

    // 当前 token 的全局位置 = cache 更新后的最后一行（解码主路径约定）。
    let pos = kv_len - 1;
    let win_start = pos.saturating_sub(cfg.window);
    // 块 b 覆盖的 token 列区间（[b·bs, (b+1)·bs)，且不越过已写长度）。
    let block_cols = |b: usize| b * bs..(b * bs + bs).min(kv_len);

    let mut out_parts = Vec::with_capacity(heads_kv);
    for g in 0..heads_kv {
        let h0 = g * n_rep;
        let h1 = h0 + n_rep;

        // ③a 组内并集列（升序去重）
        let mut cols: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for selected in &topk_per_head[h0..h1] {
            for &b in selected {
                cols.extend(block_cols(b));
            }
        }
        cols.extend(win_start..=pos);
        let cols: Vec<usize> = cols.into_iter().collect();
        let l = cols.len();
        assert!(
            l > 0,
            "KV 头组 {g} 可见列为空（kv_len={kv_len}，top_k={top_k}）"
        );

        // ③b 从全量 K/V 按列 gather（单 KV 头 → select token 列）→ [1,1,L,d]
        let idx: Vec<i32> = cols.iter().map(|&c| c as i32).collect();
        let idx_t = Tensor::<1, Int>::from_data(TensorData::new(idx, [l]), &device);
        let k_single = k
            .clone()
            .slice([0..batch, g..g + 1, 0..kv_len, 0..head_dim]);
        let v_single = v
            .clone()
            .slice([0..batch, g..g + 1, 0..kv_len, 0..head_dim]);
        let k_g = repeat_kv(k_single.select(2, idx_t.clone()), n_rep);
        let v_g = repeat_kv(v_single.select(2, idx_t), n_rep);

        // ④ per-head 可见性 mask（列级判断，同页路径）
        let mut mask_flat = Vec::with_capacity(n_rep * l);
        for selected in &topk_per_head[h0..h1] {
            let in_block = |c: usize| selected.iter().any(|&b| block_cols(b).contains(&c));
            for &c in &cols {
                let visible = c >= win_start || in_block(c);
                mask_flat.push(!visible);
            }
        }
        let mask_t: Tensor<4, Bool> =
            Tensor::from_data(TensorData::new(mask_flat, [1, n_rep, 1, l]), &device);

        let q_g = q.clone().slice([0..batch, h0..h1, 0..sq, 0..head_dim]);
        let out_g = scaled_dot_product_attention(q_g, k_g, v_g, scale, Some(mask_t));
        out_parts.push(out_g);
    }
    Tensor::cat(out_parts, 1) // [1, Hq, 1, d]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::get_device;
    use crate::model::sparse::sparse_attention_mask;

    /// 辅助：造一个 decode 场景的 cache（逐 token update），返回 (cache, k, v)。
    /// `k/v` 为 `[1, Hkv, tokens, D]` 的完整随机张量（作参考/对照用）。
    fn make_cache(
        heads_kv: usize,
        head_dim: usize,
        tokens: usize,
    ) -> (KVCache, Tensor<4>, Tensor<4>) {
        let device = get_device();
        let mut cache = KVCache::new();
        let k = Tensor::<4>::random(
            [1, heads_kv, tokens, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let v = Tensor::<4>::random(
            [1, heads_kv, tokens, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        for i in 0..tokens {
            cache.update(
                k.clone().slice([0..1, 0..heads_kv, i..i + 1, 0..head_dim]),
                v.clone().slice([0..1, 0..heads_kv, i..i + 1, 0..head_dim]),
            );
        }
        (cache, k, v)
    }

    /// 参考实现（mask 位）：q（head 布局）对全量 k/v（kv 布局，重复后）打分。
    fn mask_reference(
        q: &Tensor<4>,
        k_full: &Tensor<4>,
        v_full: &Tensor<4>,
        n_rep: usize,
        cfg: &SparseAttentionConfig,
        offset: usize,
    ) -> Tensor<4> {
        let [_, _, _, head_dim] = q.dims();
        let k_rep = repeat_kv(k_full.clone(), n_rep);
        let v_rep = repeat_kv(v_full.clone(), n_rep);
        let mask = sparse_attention_mask(q.clone(), k_rep.clone(), offset, cfg);
        let scale = (head_dim as f32).sqrt().recip();
        scaled_dot_product_attention(q.clone(), k_rep, v_rep, scale, Some(mask))
    }

    /// 页路径（cache 数据源，block_size == page_size == 16）≡ mask 参考。
    #[test]
    fn test_paged_matches_mask_reference() {
        let device = get_device();
        let heads_q = 4;
        let heads_kv = 2;
        let head_dim = 8;
        let tokens = 24; // → 2 页（16 + 8，跨页末页偏置）
        let (cache, k, v) = make_cache(heads_kv, head_dim, tokens);

        let q = Tensor::<4>::random(
            [1, heads_q, 1, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let cfg = SparseAttentionConfig::new(16, 2, 8); // bs == 页大小 → 页路径
        let n_rep = heads_q / heads_kv;

        let out_paged = sparse_decode_attention_cached(q.clone(), &cache, n_rep, &cfg);
        let out_ref = mask_reference(&q, &k, &v, n_rep, &cfg, tokens - 1);

        assert_eq!(out_ref.dims(), out_paged.dims());
        let diff = (out_ref - out_paged).abs().max().into_scalar::<f32>();
        println!("paged vs mask max diff: {diff}");
        assert!(diff < 1e-4, "页路径应与 mask 位一致，实际 diff = {diff}");
    }

    /// 非对齐（bs=4 ≠ 页 16）→ 自动回退 get_full 路径，仍与 mask 参考一致。
    #[test]
    fn test_fallback_path_matches_mask_reference() {
        let device = get_device();
        let heads_q = 4;
        let heads_kv = 2;
        let head_dim = 8;
        let tokens = 24;
        let (cache, k, v) = make_cache(heads_kv, head_dim, tokens);

        let q = Tensor::<4>::random(
            [1, heads_q, 1, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let cfg = SparseAttentionConfig::new(4, 2, 8); // bs=4 ≠ 16 → 回退
        let n_rep = heads_q / heads_kv;

        let out_fb = sparse_decode_attention_cached(q.clone(), &cache, n_rep, &cfg);
        let out_ref = mask_reference(&q, &k, &v, n_rep, &cfg, tokens - 1);

        let diff = (out_ref - out_fb).abs().max().into_scalar::<f32>();
        println!("fallback vs mask max diff: {diff}");
        assert!(diff < 1e-4, "回退路径应与 mask 位一致，实际 diff = {diff}");
    }

    /// 页路径与全量拼接入口（同 cache、同 cfg）直接互相一致（多页场景）。
    #[test]
    fn test_paged_matches_full_tensor_entry() {
        let device = get_device();
        let heads_q = 4;
        let heads_kv = 2;
        let head_dim = 8;
        let tokens = 48; // → 3 页（16 + 16 + 16，恰好全满页）
        let (cache, _, _) = make_cache(heads_kv, head_dim, tokens);

        let q = Tensor::<4>::random(
            [1, heads_q, 1, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let cfg = SparseAttentionConfig::new(16, 3, 16);
        let n_rep = heads_q / heads_kv;

        let out_paged = sparse_decode_attention_cached(q.clone(), &cache, n_rep, &cfg);
        let (k_full, v_full) = cache.get_full();
        let out_full = sparse_decode_attention(q, k_full, v_full, n_rep, &cfg);

        let diff = (out_full - out_paged).abs().max().into_scalar::<f32>();
        println!("paged vs full-entry max diff: {diff}");
        assert!(diff < 1e-4, "页路径应与全量入口一致，实际 diff = {diff}");
    }

    /// 核心等价性（回归保护）：全量入口 ≡ mask 参考。
    #[test]
    fn test_compact_matches_mask_reference() {
        let device = get_device();
        let heads_q = 4;
        let heads_kv = 2;
        let head_dim = 8;
        let kv_len = 40;

        let q = Tensor::<4>::random(
            [1, heads_q, 1, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let k = Tensor::<4>::random(
            [1, heads_kv, kv_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let v = Tensor::<4>::random(
            [1, heads_kv, kv_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let cfg = SparseAttentionConfig::new(4, 2, 8);
        let n_rep = heads_q / heads_kv;

        let out_compact = sparse_decode_attention(q.clone(), k.clone(), v.clone(), n_rep, &cfg);
        let out_ref = mask_reference(&q, &k, &v, n_rep, &cfg, kv_len - 1);

        assert_eq!(out_ref.dims(), out_compact.dims());
        let diff = (out_ref - out_compact).abs().max().into_scalar::<f32>();
        println!("compact vs mask max diff: {diff}");
        assert!(
            diff < 1e-4,
            "紧凑执行位应与 mask 位一致，实际 diff = {diff}"
        );
    }

    /// 全窗口覆盖（window >= kv_len）时，紧凑位退化为 dense。
    #[test]
    fn test_full_window_equals_dense() {
        let device = get_device();
        let heads_q = 2;
        let heads_kv = 1;
        let head_dim = 8;
        let kv_len = 16;

        let q = Tensor::<4>::random(
            [1, heads_q, 1, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let k = Tensor::<4>::random(
            [1, heads_kv, kv_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let v = Tensor::<4>::random(
            [1, heads_kv, kv_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let cfg = SparseAttentionConfig::new(4, 2, kv_len); // 窗口盖满全部历史
        let n_rep = heads_q / heads_kv;

        let k_rep = repeat_kv(k.clone(), n_rep);
        let v_rep = repeat_kv(v.clone(), n_rep);
        let scale = (head_dim as f32).sqrt().recip();
        let out_dense = scaled_dot_product_attention(q.clone(), k_rep, v_rep, scale, None);

        let out_compact = sparse_decode_attention(q, k, v, n_rep, &cfg);

        let diff = (out_dense - out_compact).abs().max().into_scalar::<f32>();
        println!("full-window vs dense max diff: {diff}");
        assert!(
            diff < 1e-4,
            "窗口全覆盖时紧凑位应等于 dense，实际 diff = {diff}"
        );
    }

    /// 早期长文（kv 很短、块数不足 top-k）不 panic，且输出有限。
    #[test]
    fn test_short_history_does_not_panic() {
        let device = get_device();
        let heads_q = 2;
        let heads_kv = 2;
        let head_dim = 8;
        let kv_len = 3; // < block_size → 单块，且 top-k 被夹到 1

        let q = Tensor::<4>::random(
            [1, heads_q, 1, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let k = Tensor::<4>::random(
            [1, heads_kv, kv_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let v = Tensor::<4>::random(
            [1, heads_kv, kv_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let cfg = SparseAttentionConfig::new(8, 4, 2); // block=8 > kv=3
        let out = sparse_decode_attention(q, k, v, heads_q / heads_kv, &cfg);
        let finite = out.clone().abs().sum().into_scalar::<f32>();
        assert!(finite.is_finite(), "短历史输出应有限");
    }

    /// gather_spans 原语自测：页读取结果 == 全量张量对应切片（含跨页、末页）。
    #[test]
    fn test_gather_spans_matches_full_slice() {
        let heads_kv = 2;
        let head_dim = 8;
        let tokens = 40; // 2 页满 + 8（末页短）
        let (cache, k, v) = make_cache(heads_kv, head_dim, tokens);

        // 跨页 + 末页区间：[2,20) ∪ [35, 40)
        let spans = [(2usize, 20usize), (35, 40)];
        let (kg, vg) = cache.gather_spans(None, &spans);
        let kr = Tensor::cat(
            vec![
                k.clone().slice([0..1, 0..heads_kv, 2..20, 0..head_dim]),
                k.clone().slice([0..1, 0..heads_kv, 35..40, 0..head_dim]),
            ],
            2,
        );
        let vr = Tensor::cat(
            vec![
                v.clone().slice([0..1, 0..heads_kv, 2..20, 0..head_dim]),
                v.clone().slice([0..1, 0..heads_kv, 35..40, 0..head_dim]),
            ],
            2,
        );
        let dk = (kg - kr).abs().max().into_scalar::<f32>();
        let dv = (vg - vr).abs().max().into_scalar::<f32>();
        assert!(dk < 1e-4, "gather_spans K 应与全量切片一致，diff = {dk}");
        assert!(dv < 1e-4, "gather_spans V 应与全量切片一致，diff = {dv}");

        // 单头版本（head=1）与全头第 1 头切片一致
        let (kg1, vg1) = cache.gather_spans(Some(1), &[(2usize, 20usize)]);
        let kr1 = k.slice([0..1, 1..2, 2..20, 0..head_dim]);
        let vr1 = v.slice([0..1, 1..2, 2..20, 0..head_dim]);
        let dk1 = (kg1 - kr1).abs().max().into_scalar::<f32>();
        let dv1 = (vg1 - vr1).abs().max().into_scalar::<f32>();
        assert!(dk1 < 1e-4, "单头 gather_spans K 应一致，diff = {dk1}");
        assert!(dv1 < 1e-4, "单头 gather_spans V 应一致，diff = {dv1}");
    }

    /// head_rows 原语自测：页首行 == 全量 K 的 0,16,32 行（对齐场景）。
    #[test]
    fn test_head_rows_matches_block_starts() {
        let heads_kv = 2;
        let head_dim = 8;
        let tokens = 40; // 3 页（16,16,8）→ 首行 token 0,16,32
        let (cache, k, _) = make_cache(heads_kv, head_dim, tokens);

        let rows = cache.head_rows().unwrap();
        let starts = [0usize, 16, 32];
        let expected = Tensor::cat(
            starts
                .iter()
                .map(|&s| k.clone().slice([0..1, 0..heads_kv, s..s + 1, 0..head_dim]))
                .collect(),
            2,
        );
        let diff = (rows - expected).abs().max().into_scalar::<f32>();
        assert!(diff < 1e-4, "head_rows 应等于各页首行，diff = {diff}");
    }
}
