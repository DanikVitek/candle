//! Traits to Define Backend Behavior
//!
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};

#[cfg(feature = "iex")]
use iex::iex;

pub trait BackendStorage: Sized {
    type Device: BackendDevice;

    #[cfg_attr(feature = "iex", iex)]
    fn try_clone(&self, _: &Layout) -> Result<Self>;

    fn dtype(&self) -> DType;

    fn device(&self) -> &Self::Device;

    // Maybe this should return a Cow instead so that no copy is done on the cpu case.
    #[cfg_attr(feature = "iex", iex)]
    fn to_cpu_storage(&self) -> Result<CpuStorage>;

    #[cfg_attr(feature = "iex", iex)]
    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn powf(&self, _: &Layout, _: f64) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn elu(&self, _: &Layout, _: f64) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()>;

    #[cfg_attr(feature = "iex", iex)]
    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()>;

    #[cfg_attr(feature = "iex", iex)]
    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self>;

    #[cfg_attr(feature = "iex", iex)]
    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()>;

    // Similar to cudaMemcpy2D, though values are in elements and not in bytes.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(feature = "iex", iex)]
    fn copy2d(
        &self,
        _: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_stride1: usize,
        _dst_stride1: usize,
        _src_offset: usize,
        _dst_offset: usize,
    ) -> Result<()>;

    #[cfg_attr(feature = "iex", iex)]
    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()>;
}

pub trait BackendDevice: Sized + std::fmt::Debug + Clone {
    type Storage: BackendStorage;

    // TODO: Make the usize generic and part of a generic DeviceLocation.
    #[cfg_attr(feature = "iex", iex)]
    fn new(_: usize) -> Result<Self>;

    fn location(&self) -> crate::DeviceLocation;

    fn same_device(&self, _: &Self) -> bool;

    #[cfg_attr(feature = "iex", iex)]
    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    /// # Safety
    /// This function is unsafe as it doesn't initialize the underlying data store.
    /// The caller should ensure that the data is properly initialized as early as possible
    /// after this call.
    #[cfg_attr(feature = "iex", iex)]
    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    #[cfg_attr(feature = "iex", iex)]
    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage>;

    #[cfg_attr(feature = "iex", iex)]
    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage>;

    #[cfg_attr(feature = "iex", iex)]
    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage>;

    #[cfg_attr(feature = "iex", iex)]
    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage>;

    #[cfg_attr(feature = "iex", iex)]
    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage>;

    #[cfg_attr(feature = "iex", iex)]
    fn set_seed(&self, _: u64) -> Result<()>;

    #[cfg_attr(feature = "iex", iex)]
    fn get_current_seed(&self) -> Result<u64>;

    /// Synchronize should block until all the operations on the device are completed.
    #[cfg_attr(feature = "iex", iex)]
    fn synchronize(&self) -> Result<()>;
}
