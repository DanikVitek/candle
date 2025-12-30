pub(crate) mod affine;
pub(crate) mod binary;
pub(crate) mod broadcast;
pub(crate) mod conv_transpose2d;
pub(crate) mod copy;
pub(crate) mod matmul;
pub(crate) mod qmatmul;
pub(crate) mod random;
pub(crate) mod reduce;
pub(crate) mod unary;
pub(crate) mod where_cond;

use candle_core::{Device, Result};
#[cfg(feature = "iex")]
use iex::iex;

pub(crate) trait BenchDevice {
    #[cfg_attr(feature = "iex", iex)]
    fn sync(&self) -> Result<()>;

    fn bench_name<S: Into<String>>(&self, name: S) -> String;
}

impl BenchDevice for Device {
    #[cfg_attr(feature = "iex", iex)]
    fn sync(&self) -> Result<()> {
        match self {
            Device::Cpu => Ok(()),
            Device::Cuda(device) => {
                #[cfg(feature = "cuda")]
                {
                    use candle_core::backend::BackendDevice;
                    return Ok(device.synchronize()?);
                }
                #[cfg(not(feature = "cuda"))]
                panic!("Cuda device without cuda feature enabled: {device:?}")
            }
            Device::Metal(device) => {
                #[cfg(feature = "metal")]
                return device.wait_until_completed();
                #[cfg(not(feature = "metal"))]
                panic!("Metal device without metal feature enabled: {device:?}")
            }
        }
    }

    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        match self {
            Device::Cpu => {
                let cpu_type = if cfg!(feature = "accelerate") {
                    "accelerate"
                } else if cfg!(feature = "mkl") {
                    "mkl"
                } else {
                    "cpu"
                };
                format!("{}_{}", cpu_type, name.into())
            }
            Device::Cuda(_) => format!("cuda_{}", name.into()),
            Device::Metal(_) => format!("metal_{}", name.into()),
        }
    }
}

struct BenchDeviceHandler {
    devices: Vec<Device>,
}

impl BenchDeviceHandler {
    #[cfg_attr(feature = "iex", iex)]
    pub fn new() -> Result<Self> {
        let mut devices = Vec::new();
        if cfg!(feature = "metal") {
            devices.push(Device::new_metal(0)?);
        } else if cfg!(feature = "cuda") {
            devices.push(Device::new_cuda(0)?);
        } else {
            devices.push(Device::Cpu);
        }
        Ok(Self { devices })
    }
}
