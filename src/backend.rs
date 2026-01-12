use burn::backend::Autodiff;

#[cfg(feature = "wgpu")]
mod wgpu_backend {
    use super::*;
    use burn_wgpu::{Wgpu, WgpuDevice};

    pub type MyBackend = Wgpu;
    pub type MyAutodiffBackend = Autodiff<Wgpu>;
    pub type MyDevice = WgpuDevice;

    pub fn get_device() -> MyDevice {
        WgpuDevice::default()
    }
}

#[cfg(feature = "cuda")]
mod cuda_backend {
    use super::*;
    use burn_tch::{LibTorch, LibTorchDevice};

    pub type MyBackend = LibTorch;
    pub type MyAutodiffBackend = Autodiff<LibTorch>;
    pub type MyDevice = LibTorchDevice;

    pub fn get_device() -> MyDevice {
        LibTorchDevice::Cuda(0)
    }
}

#[cfg(feature = "wgpu")]
pub use wgpu_backend::*;

#[cfg(feature = "cuda")]
pub use cuda_backend::*;
