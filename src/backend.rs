use burn::tensor::Device;
use clap::ValueEnum;
use std::sync::OnceLock;

#[cfg(not(any(
    feature = "cpu",
    feature = "cuda",
    feature = "rocm",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu",
    feature = "wgpu",
    feature = "cubecl-cpu"
)))]
compile_error!("Enable a MetaI compute backend in Cargo");

/// Compiled runtimes, in automatic selection order. The CLI may override this choice.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Runtime {
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "rocm")]
    Rocm,
    #[cfg(feature = "metal")]
    Metal,
    #[cfg(feature = "vulkan")]
    Vulkan,
    #[cfg(feature = "webgpu")]
    Webgpu,
    #[cfg(feature = "wgpu")]
    Wgpu,
    #[cfg(feature = "cubecl-cpu")]
    CubeclCpu,
    #[cfg(feature = "cpu")]
    Flex,
}

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

pub fn select(runtime: Runtime) -> anyhow::Result<()> {
    RUNTIME
        .set(runtime)
        .map_err(|_| anyhow::anyhow!("compute runtime already selected"))
}

pub fn runtime() -> Runtime {
    *RUNTIME.get_or_init(|| Runtime::value_variants()[0])
}

pub fn get_device() -> Device {
    device_for_rank(0, 1)
}

pub fn device_for_rank(rank: usize, world_size: usize) -> Device {
    assert!(rank < world_size, "device rank exceeds topology");
    #[cfg(any(
        feature = "wgpu",
        feature = "vulkan",
        feature = "webgpu",
        feature = "metal"
    ))]
    let adapter = || {
        use burn::tensor::DeviceKind;
        if world_size == 1 {
            DeviceKind::default()
        } else if cfg!(target_os = "macos") {
            DeviceKind::IntegratedGpu(rank)
        } else {
            DeviceKind::DiscreteGpu(rank)
        }
    };
    match runtime() {
        #[cfg(feature = "cuda")]
        Runtime::Cuda => Device::cuda(rank),
        #[cfg(feature = "rocm")]
        Runtime::Rocm => Device::rocm(rank),
        #[cfg(feature = "wgpu")]
        Runtime::Wgpu => Device::wgpu(adapter()),
        #[cfg(feature = "metal")]
        Runtime::Metal => Device::metal(adapter()),
        #[cfg(feature = "vulkan")]
        Runtime::Vulkan => Device::vulkan(adapter()),
        #[cfg(feature = "webgpu")]
        Runtime::Webgpu => Device::webgpu(adapter()),
        #[cfg(feature = "cubecl-cpu")]
        Runtime::CubeclCpu => Device::cpu(),
        #[cfg(feature = "cpu")]
        Runtime::Flex => Device::flex(),
    }
}
