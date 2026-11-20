use metai::backend::get_device;
use metai::model::config::MetaIConfig;
use metai::model::MetaIModel;
use metai::{AttentionVariant, MlaConfig, NormVariant, RouterVariant, SparseAttentionConfig};

#[test]
fn test_model_forward() {
    let device = get_device();
    let config = MetaIConfig::tiny();
    let model = MetaIModel::new(&config, 0, &device);

    let tokens = burn::tensor::Tensor::<2, burn::tensor::Int>::from_data(
        burn::tensor::TensorData::new([1i32, 2, 3, 4, 5].to_vec(), [1, 5]),
        &device,
    );

    let output = model.forward(tokens, None, None);
    assert_eq!(output.dims(), [1, 5, config.vocab_size]);
}

fn sample_tokens(device: &burn::tensor::Device) -> burn::tensor::Tensor<2, burn::tensor::Int> {
    burn::tensor::Tensor::from_data(
        burn::tensor::TensorData::new([1i32, 2, 3, 4, 5].to_vec(), [1, 5]),
        device,
    )
}

/// 归一化族三变体均可装配并前向（默认 RMSNorm 行为已在 test_model_forward 覆盖）。
#[test]
fn test_norm_variants_forward() {
    let device = get_device();
    for variant in [
        NormVariant::RmsNorm,
        NormVariant::LayerNorm,
        NormVariant::DyT,
    ] {
        let mut config = MetaIConfig::tiny();
        config.norm = variant;
        let model = MetaIModel::new(&config, 0, &device);
        let output = model.forward(sample_tokens(&device), None, None);
        assert_eq!(output.dims(), [1, 5, config.vocab_size], "{variant:?}");
    }
}

/// KV 侧稀疏注意力（block top-k + window，mask 位）在整模型装配路径可用。
#[test]
fn test_sparse_forward() {
    let device = get_device();
    let mut config = MetaIConfig::tiny();
    config.sparse = Some(SparseAttentionConfig::new(2, 2, 2));
    let model = MetaIModel::new(&config, 0, &device);

    let output = model.forward(sample_tokens(&device), None, None);
    assert_eq!(output.dims(), [1, 5, config.vocab_size]);
}

/// MLA（低秩 latent 注意力）在整模型装配路径可用（默认 dense）。
#[test]
fn test_mla_forward() {
    let device = get_device();
    let mut config = MetaIConfig::tiny();
    config.attn = AttentionVariant::Mla(MlaConfig::new(16, 8));
    let model = MetaIModel::new(&config, 0, &device);

    let output = model.forward(sample_tokens(&device), None, None);
    assert_eq!(output.dims(), [1, 5, config.vocab_size]);
}

/// MLA 整模型端到端：prefill（整段 causal）≡ 逐 token decode（cache）。
/// 该测试驱动「latent 单通道 cache + 共享 rope 旋转」贯穿全部层与 RoPE 装配。
#[test]
fn test_mla_prefill_matches_decode() {
    use burn::tensor::Tensor;
    use metai::infer::cache::ModelCache;

    let device = get_device();
    let mut config = MetaIConfig::tiny();
    config.attn = AttentionVariant::Mla(MlaConfig::new(16, 8));
    let model = MetaIModel::new(&config, 0, &device);

    let seq_len = 9;
    let input = Tensor::<2, burn::tensor::Int>::random(
        [1, seq_len],
        burn::tensor::Distribution::Uniform(1.0, 100.0),
        &device,
    );

    // 1. prefill：整段（无 cache）
    let out_prefill = model.forward(input.clone(), None, None);

    // 2. decode：逐 token 走 cache
    let mut cache = ModelCache::new(model.blocks.len());
    let mut parts = Vec::new();
    for i in 0..seq_len {
        let step = input.clone().slice([0..1, i..i + 1]);
        parts.push(model.forward(step, None, Some(&mut cache)));
    }
    let out_decode = Tensor::cat(parts, 1);

    let diff = (out_prefill - out_decode).abs().max().into_scalar::<f32>();
    println!("MLA full-model prefill vs decode Max Diff: {diff}");
    assert!(diff < 1e-4, "MLA 全模型 prefill 应 ≡ decode，diff = {diff}");
}

/// MLA 权重在 autodiff 后端可回传梯度（训练路径可用）。
#[test]
fn test_mla_gradients() {
    use metai::model::attention::Attention;

    let device = get_device().autodiff();
    let mut config = MetaIConfig::tiny();
    config.attn = AttentionVariant::Mla(MlaConfig::new(16, 8));
    let model = MetaIModel::new(&config, 0, &device);

    let logits = model.forward(sample_tokens(&device), None, None);
    let loss = logits.sum();
    let grads = loss.backward();

    // q / latent 压缩 / rope key / 输出四组权重都应参与前向 → 有梯度
    let mla = match &model.blocks[0].attention {
        Attention::Mla(m) => m,
        _ => unreachable!("config.attn 为 MLA，第一个块必为 MLA"),
    };
    let check = |w: burn::tensor::Tensor<2>| -> bool { w.grad(&grads).is_some() };
    assert!(check(mla.w_q_nope.weight.val()), "w_q_nope 应有梯度");
    assert!(
        check(mla.w_dkv.weight.val()),
        "w_dkv（latent 压缩）应有梯度"
    );
    assert!(check(mla.w_k_rope.weight.val()), "w_k_rope 应有梯度");
    assert!(check(mla.w_o.weight.val()), "w_o 应有梯度");
}

/// MLA × KV 稀疏组合在装配期被显式拒绝（DeepSeek-V3.2 路线为后续任务）。
#[test]
#[should_panic(expected = "MLA 与 config.sparse")]
fn test_mla_with_sparse_rejected() {
    let device = get_device();
    let mut config = MetaIConfig::tiny();
    config.attn = AttentionVariant::Mla(MlaConfig::new(16, 8));
    config.sparse = Some(SparseAttentionConfig::new(2, 2, 2));
    let _ = MetaIModel::new(&config, 0, &device);
}

/// 路由族两变体均可装配并前向。
#[test]
fn test_router_variants_forward() {
    let device = get_device();
    for variant in [RouterVariant::SoftmaxTopK, RouterVariant::SigmoidTopK] {
        let mut config = MetaIConfig::tiny();
        config.router = variant;
        let model = MetaIModel::new(&config, 0, &device);
        let output = model.forward(sample_tokens(&device), None, None);
        assert_eq!(output.dims(), [1, 5, config.vocab_size], "{variant:?}");
    }
}

/// 变体组合（DyT + SigmoidTopK）在 autodiff 后端梯度可回传。
#[test]
fn test_variant_gradients() {
    let device = get_device().autodiff();
    let mut config = MetaIConfig::tiny();
    config.norm = NormVariant::DyT;
    config.router = RouterVariant::SigmoidTopK;
    let model = MetaIModel::new(&config, 0, &device);

    let logits = model.forward(sample_tokens(&device), None, None);
    let loss = logits.sum();
    let grads = loss.backward();

    // DyT 的 alpha 是新增可学习参数（形状 [1]），一定参与前向 → 应能取到梯度
    let alpha = match &model.blocks[0].attn_norm {
        metai::model::norm::Norm::DyT(dyt) => dyt.alpha.val(),
        _ => unreachable!("config.norm 为 DyT，第一个块的 attn_norm 必为 DyT"),
    };
    let alpha_grad = alpha.grad(&grads);
    assert!(alpha_grad.is_some(), "DyT alpha 应产生梯度");

    // Sigmoid 路由经 gate 反向，gate 权重应能取到梯度
    let gate = model.blocks[0].mlp.gate.weight.val();
    let gate_grad = gate.grad(&grads);
    assert!(gate_grad.is_some(), "Sigmoid 路由 gate 应产生梯度");
}

#[test]
fn one_softmax_expert_matches_direct_ffn_and_gradients() {
    use burn::tensor::Tensor;
    use metai::model::moe::MoE;
    let d = get_device().autodiff();
    let moe = MoE::new(4, 8, 1, 1, 0, &d);
    let x =
        Tensor::<2>::from_floats([[0.1, 0.2, -0.1, 1.0], [0.3, 0.0, 0.2, -1.0]], &d).require_grad();
    let direct = moe.experts[0].forward(x.clone());
    let routed = moe.forward(x.clone());
    assert!(
        (direct.clone() - routed.clone())
            .abs()
            .max()
            .into_scalar::<f32>()
            < 1e-6
    );
    let direct_grad = direct.sum().backward();
    let routed_grad = routed.sum().backward();
    assert!(
        (x.grad(&direct_grad).unwrap() - x.grad(&routed_grad).unwrap())
            .abs()
            .max()
            .into_scalar::<f32>()
            < 1e-6
    );
    assert_eq!(
        moe.gate
            .weight
            .val()
            .grad(&routed_grad)
            .unwrap()
            .abs()
            .max()
            .into_scalar::<f32>(),
        0.
    );
}
