#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[cfg(feature = "iex")]
use iex::iex;

#[derive(Debug, Clone)]
pub struct MetalDevice;

#[derive(Debug)]
pub struct MetalStorage;

#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for MetalError {
    fn from(e: String) -> Self {
        MetalError::Message(e)
    }
}

macro_rules! fail {
    () => {
        unimplemented!("metal support has not been enabled, add `metal` feature to enable.")
    };
}

impl crate::backend::BackendStorage for MetalStorage {
    type Device = MetalDevice;

    #[cfg_attr(feature = "iex", iex)]
    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> &Self::Device {
        fail!()
    }

    #[cfg_attr(feature = "iex", iex)]
    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }
    #[cfg_attr(feature = "iex", iex)]
    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }
}

impl crate::backend::BackendDevice for MetalDevice {
    type Storage = MetalStorage;

    #[cfg_attr(feature = "iex", iex)]
    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn set_seed(&self, _: u64) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn get_current_seed(&self) -> Result<u64> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }

    fn same_device(&self, _: &Self) -> bool {
        fail!()
    }

    #[cfg_attr(feature = "iex", iex)]
    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
