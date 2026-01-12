use metai::backend::{get_device, MyBackend};
use metai::model::config::MetaIConfig;
use metai::model::MetaIModel;

#[test]
fn test_model_forward() {
    let device = get_device();
    let config = MetaIConfig::small();
    let model = MetaIModel::<MyBackend>::new(&config, 0, &device);

    let tokens = burn::tensor::Tensor::<MyBackend, 2, burn::tensor::Int>::from_data(
        burn::tensor::TensorData::new([1i32, 2, 3, 4, 5].to_vec(), [1, 5]),
        &device,
    );

    let output = model.forward(tokens, None, None);
    assert_eq!(output.dims(), [1, 5, config.vocab_size]);
}
