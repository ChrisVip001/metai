#[cfg(feature = "wgpu")]
mod wgpu_backend {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::backend::Autodiff;

    pub type MyBackend = Wgpu;
    pub type MyAutodiffBackend = Autodiff<Wgpu>;
    pub type MyDevice = WgpuDevice;

    pub fn get_device() -> MyDevice {
        WgpuDevice::default()
    }
}

#[cfg(feature = "cuda")]
mod cuda_backend {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    use burn::backend::Autodiff;

    // Use float16 by default for performance on modern GPUs
    pub type MyBackend = LibTorch<f32>;
    pub type MyAutodiffBackend = Autodiff<LibTorch<f32>>;
    pub type MyDevice = LibTorchDevice;

    pub fn get_device() -> MyDevice {
        #[cfg(not(target_os = "macos"))]
        return LibTorchDevice::Cuda(0);

        #[cfg(target_os = "macos")]
        return LibTorchDevice::Mps;
    }
}

#[cfg(feature = "wgpu")]
pub use wgpu_backend::*;

#[cfg(feature = "cuda")]
pub use cuda_backend::*;
