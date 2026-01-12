use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::model::positional_encoding::RoPE;

#[derive(Module, Debug)]
pub struct GroupedQueryAttention<B: Backend> {
    pub w_q: Linear<B>,
    pub w_k: Linear<B>,
    pub w_v: Linear<B>,
    pub w_o: Linear<B>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> GroupedQueryAttention<B> {
    pub fn new(hidden_dim: usize, n_heads: usize, n_kv_heads: usize, device: &B::Device) -> Self {
        let head_dim = hidden_dim / n_heads;
        Self {
            w_q: LinearConfig::new(hidden_dim, n_heads * head_dim)
                .with_bias(false)
                .init(device),
            w_k: LinearConfig::new(hidden_dim, n_kv_heads * head_dim)
                .with_bias(false)
                .init(device),
            w_v: LinearConfig::new(hidden_dim, n_kv_heads * head_dim)
                .with_bias(false)
                .init(device),
            w_o: LinearConfig::new(n_heads * head_dim, hidden_dim)
                .with_bias(false)
                .init(device),
            n_heads,
            n_kv_heads,
            head_dim,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RoPE<B>,
        mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
        cache: Option<&mut crate::infer::cache::KVCache<B>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden_dim] = x.dims();

        // 1. Projection
        let q = self.w_q.forward(x.clone());
        let k = self.w_k.forward(x.clone());
        let v = self.w_v.forward(x);

        // 2. Reshape [Batch, Seq, Heads * Dim] -> [Batch, Heads, Seq, Dim]
        let q = q
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // 3. 应用 RoPE
        let offset = cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);
        let q = rope.forward_with_offset(q, offset);
        let k = rope.forward_with_offset(k, offset);

        if let Some(cache) = cache {
            // Update Cache
            cache.update(k, v);

            // Fetch Full Cache (Concat of all blocks)
            // Ideally we should use PagedAttention kernel, but for portability/simplicity on WGPU/M1:
            // We concat to standard tensors.
            let (k_full, v_full) = cache.get_full();

            // GQA Repeat info
            let n_rep = self.n_heads / self.n_kv_heads;

            // Repeat Heads for GQA
            let k = self.repeat_heads(k_full, n_rep);
            let v = self.repeat_heads(v_full, n_rep);

            // Compute Attention Scores
            let scale = (self.head_dim as f32).sqrt().recip();
            let attn_scores = q.matmul(k.swap_dims(2, 3)) * scale;

            // Softmax
            let attn_probs = burn::tensor::activation::softmax(attn_scores, 3);

            // Output
            let out = attn_probs.matmul(v);

            let out =
                out.swap_dims(1, 2)
                    .reshape([batch_size, seq_len, self.n_heads * self.head_dim]);
            self.w_o.forward(out)
        } else {
            // --- Standard Non-Cached Attention ---

            let n_rep = self.n_heads / self.n_kv_heads;
            let k = self.repeat_heads(k, n_rep);
            let v = self.repeat_heads(v, n_rep);

            let device = q.device();
            let scale = (self.head_dim as f32).sqrt().recip();
            let mut attn_scores = q.matmul(k.swap_dims(2, 3)) * scale;

            // Causal Mask
            attn_scores = if seq_len > 1 {
                let causal_mask = Tensor::<B, 2>::ones([seq_len, seq_len], &device)
                    .triu(1)
                    .bool()
                    .reshape([1, 1, seq_len, seq_len]);
                attn_scores.mask_fill(causal_mask, -f32::INFINITY)
            } else {
                attn_scores
            };

            if let Some(pad_mask) = mask {
                if seq_len > 1 {
                    let pad_mask_expanded = pad_mask
                        .unsqueeze::<4>()
                        .reshape([batch_size, 1, 1, seq_len]);
                    attn_scores = attn_scores.mask_fill(pad_mask_expanded, -f32::INFINITY);
                }
            }

            let attn = burn::tensor::activation::softmax(attn_scores, 3);
            let out = attn.matmul(v);

            let out =
                out.swap_dims(1, 2)
                    .reshape([batch_size, seq_len, self.n_heads * self.head_dim]);
            self.w_o.forward(out)
        }
    }

    fn repeat_heads(&self, x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        if n_rep == 1 {
            return x;
        }

        let [batch_size, n_kv_heads, seq_len, head_dim] = x.dims();

        // 使用增加维度、展开、reshape 的方式实现 repeat
        x.unsqueeze_dims::<5>(&[2])
            .expand([batch_size, n_kv_heads, n_rep, seq_len, head_dim])
            .reshape([batch_size, n_kv_heads * n_rep, seq_len, head_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{get_device, MyBackend}; // Use project defined backend
    use crate::infer::cache::KVCache;
    use crate::model::positional_encoding::RoPE;

    type TestBackend = MyBackend;

    #[test]
    fn test_paged_attention_consistency() {
        let device = get_device();
        let hidden_dim = 16;
        let n_heads = 4;
        let n_kv_heads = 2; // GQA
        let head_dim = hidden_dim / n_heads;

        let attn =
            GroupedQueryAttention::<TestBackend>::new(hidden_dim, n_heads, n_kv_heads, &device);
        let rope = RoPE::new(head_dim, 128, 10000.0, &device);

        // 1. Full Sequence Forward (Standard)
        let seq_len = 4;
        let input = Tensor::<TestBackend, 3>::random(
            [1, seq_len, hidden_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        // Standard forward (no cache)
        let out_std = attn.forward(input.clone(), &rope, None, None);

        // 2. Step-by-Step Forward (Paged Cache)
        let mut cache = KVCache::new();
        let mut out_paged_parts = Vec::new();

        for i in 0..seq_len {
            // Check output of step i
            let input_step = input.clone().slice([0..1, i..i + 1, 0..hidden_dim]);

            // Forward with cache
            let out_step = attn.forward(input_step, &rope, None, Some(&mut cache));
            out_paged_parts.push(out_step);
        }

        let out_paged = Tensor::cat(out_paged_parts, 1);

        // 3. Compare
        // We expect some floating point divergence due to different calculation order (block sum vs global softmax)
        // Wait, Global Softmax over concat scores IS mathematically equivalent.
        // But in step-by-step decoding, we only attend to past!
        // In Full Sequence Standard Forward, we use Causal Mask.
        // Causal Mask ensures pos i only attends to 0..=i.
        // In Cached Forward step i, we attend to cache which contains 0..=i.
        // So they SHOULD be equivalent.

        let diff = (out_std - out_paged).abs().max().into_scalar();
        println!("Max Diff: {}", diff);
        assert!(diff < 1e-4);
    }
}
