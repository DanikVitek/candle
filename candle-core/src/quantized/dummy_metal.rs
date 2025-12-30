#![allow(unused)]
use super::GgmlDType;
use crate::{Error, MetalDevice, MetalStorage, Result};

#[cfg(feature = "iex")]
use iex::iex;

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
}

impl QMetalStorage {
    #[cfg_attr(feature = "iex", iex)]
    pub fn zeros(_: &MetalDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    #[cfg_attr(feature = "iex", iex)]
    pub fn dequantize(&self, _elem_count: usize) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    pub fn quantize(&mut self, _src: &MetalStorage) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    pub fn quantize_imatrix(
        &mut self,
        _src: &MetalStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    pub fn quantize_onto(&mut self, _src: &crate::CpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    #[cfg_attr(feature = "iex", iex)]
    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &MetalStorage,
        _layout: &crate::Layout,
    ) -> Result<(MetalStorage, crate::Shape)> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    #[cfg_attr(feature = "iex", iex)]
    pub fn indexed_moe_forward(
        &self,
        _: &crate::Shape,
        _: &MetalStorage,
        _: &crate::Layout,
        _: &MetalStorage,
        _: &crate::Layout,
    ) -> Result<(MetalStorage, crate::Shape)> {
        Err(Error::NotCompiledWithMetalSupport)
    }
}

#[cfg_attr(feature = "iex", iex)]
pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &MetalDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithMetalSupport)
}
