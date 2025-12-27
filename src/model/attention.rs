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
        let mut v = v
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // 3. 应用 RoPE (RoPE 现在接受 4D Tensor)
        let q = rope.forward(q);
        // Note: 如果使用 Cache，RoPE 应该针对当前位置编码。
        let mut k = rope.forward(k);

        // ----------------------
        // KV Cache Logic
        // ----------------------
        if let Some(cache) = cache {
            // RoPE 之后再存入 Cache (Key 需要是 RoBeds 后的)
            let (k_cached, v_cached) = cache.update(k, v);
            k = k_cached;
            v = v_cached;
        }

        // 4. 重复 KV 头以支持 GQA
        let n_rep = self.n_heads / self.n_kv_heads;
        let k = self.repeat_heads(k, n_rep);
        let v = self.repeat_heads(v, n_rep);

        // 5. 计算注意力分数
        // q: [Batch, Heads, QSeq, Dim], k: [Batch, Heads, KSeq, Dim]
        let device = q.device();
        let scale = (self.head_dim as f32).sqrt().recip();
        let mut attn_scores = q.matmul(k.swap_dims(2, 3)) * scale;

        // 6. 注意力掩码 (Causal Mask)
        // 只有当 Query 序列长度 > 1 时才需要因果掩码 (Prefill / Training)
        // 如果 QSeq=1 (Decoding), 它应该看到所有的 K (History + Current), 无需掩码 (除了 Padding)
        attn_scores = if seq_len > 1 {
            // 生成下三角矩阵，上三角（未来信息）设为 -inf
            let causal_mask = Tensor::<B, 2>::ones([seq_len, seq_len], &device)
                .triu(1)
                .bool()
                .reshape([1, 1, seq_len, seq_len]); // Broadcastable

            attn_scores.mask_fill(causal_mask, -f32::INFINITY)
        } else {
            attn_scores
        };

        // 6.1 应用 Padding Mask
        // pad_mask shape: [Batch, QSeq]
        // 注意：Cache 模式下，Key 可能包含 Padding (如果 Batch > 1 且之前的 History 有 Pad)。
        // 简单的 Pad Mask 传入的是针对 input x 的。
        // 在 Decoding 阶段，input x 只是 current token，通常不是 Pad。
        // 所以 Decoding 阶段 Pad Mask 几乎没用，除非我们需要屏蔽 History 中的 Pad。
        // 但目前实现里 Generator 是 Batch=1 的，所以暂时不必担心 Cache 里的 Pad。
        if let Some(pad_mask) = mask {
            // 只有在非 Cache 或者是 Prefill 阶段主要生效
            if seq_len > 1 {
                let pad_mask_expanded = pad_mask
                    .unsqueeze::<4>()
                    .reshape([batch_size, 1, 1, seq_len]);
                attn_scores = attn_scores.mask_fill(pad_mask_expanded, -f32::INFINITY);
            }
        }

        // 7. Softmax + Value 加权
        let attn = burn::tensor::activation::softmax(attn_scores, 3);
        let out = attn.matmul(v);

        // 8. 合并头并投影输出
        let out = out
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.n_heads * self.head_dim]);
        self.w_o.forward(out)
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
