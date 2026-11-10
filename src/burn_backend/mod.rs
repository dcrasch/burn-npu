//! Burn `Backend` implementation for NPU.
//!
//! Platform-specific float tensor primitives:
//!
//! - **`apple`**: `NpuFloatTensor` wraps an `i32` MLTensor handle. All float ops
//!   pass handles through FFI; no data leaves the NPU between ops. Both f32 and
//!   f16 are supported, f16 being the ANE's native format.
//! - **`intel`**: `NpuFloatTensor` is `IntelFloatTensor` (`Vec<f32>` + shape).
//!   Matmul attempts OpenVINO NPU dispatch; all other ops run on CPU or delegate
//!   to burn-flex.
//! - **`qualcomm`**: `NpuFloatTensor` is `QnnFloatTensor` (`Vec<f32>` + shape).
//!   Matmul dispatches to the Hexagon NPU when a QNN SDK was present at build
//!   time; everything else runs on CPU.
//! - **no feature**: `NpuFloatTensor` is `FlexTensor` (pure CPU fallback).
//!
//! Int/Bool tensor primitives always remain `FlexTensor` (delegated to burn-flex).

extern crate alloc;

mod bool_ops;
mod ffi;
mod float_ops;
mod int_ops;
mod module_ops;
mod quantization_ops;
pub mod tensor;

use alloc::string::String;
use burn_flex::{Flex, FlexDevice, FlexQTensor, FlexTensor};
use burn_tensor::backend::{Backend, BackendTypes, DTypeUsageSet, DeviceId, DeviceOps};
// Only the apple path narrows dtype_usage; elsewhere it delegates wholesale.
#[cfg(feature = "apple")]
use burn_tensor::backend::DTypeUsage;
use burn_tensor::ops::*;
use burn_tensor::DType;

#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
pub use tensor::NpuFloatTensor;

// ---------------------------------------------------------------------------
// Type alias for the Flex backend we delegate to.
//
// `Flex` is `Flex<f32, i32>`; burn-flex only implements `Backend` for that
// default instantiation, and dispatches element types at runtime via `DType`.
// ---------------------------------------------------------------------------
pub(super) type Fx = Flex;

// ---------------------------------------------------------------------------
// NpuBurnDevice
// ---------------------------------------------------------------------------
/// Device type for the NPU burn backend. There is only one logical device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NpuBurnDevice {
    /// The default device (routes to ANE when available, falls back to CPU).
    #[default]
    Default,
}

impl DeviceOps for NpuBurnDevice {}

impl burn_tensor::backend::Device for NpuBurnDevice {
    fn from_id(_device_id: DeviceId) -> Self {
        Self::Default
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 1,
            index_id: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// NpuBurnBackend
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Default, Debug)]
pub struct NpuBurnBackend;

/// Helper: map NpuBurnDevice -> FlexDevice for forwarding.
#[inline(always)]
pub(super) fn flex_dev() -> FlexDevice {
    FlexDevice
}

// ===========================================================================
// BackendTypes
//
// The only thing that varies per platform is the float primitive: on a real NPU
// it is the vendor handle/buffer, otherwise it is a plain FlexTensor. Int, bool
// and quantized primitives always delegate to burn-flex.
// ===========================================================================
#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
impl BackendTypes for NpuBurnBackend {
    type Device = NpuBurnDevice;

    type FloatTensorPrimitive = NpuFloatTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = FlexTensor;
    // i32 matches burn-flex's default int element, so delegated int ops do not
    // need a dtype conversion on every call.
    type IntElem = i32;

    type BoolTensorPrimitive = FlexTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = FlexQTensor;
}

#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl BackendTypes for NpuBurnBackend {
    type Device = NpuBurnDevice;

    type FloatTensorPrimitive = FlexTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = FlexTensor;
    type IntElem = i32;

    type BoolTensorPrimitive = FlexTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = FlexQTensor;
}

// ===========================================================================
// Backend
//
// Shared across every feature combination: only the reported name differs.
// ===========================================================================
impl Backend for NpuBurnBackend {
    fn name(_device: &Self::Device) -> String {
        #[cfg(feature = "apple")]
        {
            String::from("Apple ANE")
        }
        #[cfg(all(feature = "intel", not(feature = "apple")))]
        {
            String::from("Intel NPU")
        }
        #[cfg(all(feature = "qualcomm", not(feature = "apple"), not(feature = "intel")))]
        {
            String::from("Qualcomm Hexagon")
        }
        #[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
        {
            String::from("CPU fallback")
        }
    }

    fn seed(_device: &Self::Device, seed: u64) {
        <Fx as Backend>::seed(&flex_dev(), seed);
    }

    /// Report what this backend can actually do, not what the CPU delegate can.
    ///
    /// On the apple path float tensors live in MLTensor, which supports f32 and
    /// f16 but not bf16 or f64. Previously this delegated wholesale to
    /// burn-flex, so burn was told bf16 and f64 were available and every such
    /// tensor was silently handled as f32 instead.
    ///
    /// Int, bool and quantized dtypes still delegate, because those primitives
    /// really are burn-flex tensors.
    fn dtype_usage(_device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        #[cfg(feature = "apple")]
        {
            match dtype {
                // fp16 is the ANE's native format; fp32 is accepted too.
                DType::F32 | DType::F16 => {
                    return DTypeUsage::Storage | DTypeUsage::Arithmetic;
                }
                DType::BF16 | DType::F64 => return DTypeUsageSet::empty(),
                _ => {}
            }
        }
        <Fx as Backend>::dtype_usage(&flex_dev(), dtype)
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }
}

// ===========================================================================
// ActivationOps (all methods have defaults)
// ===========================================================================
#[cfg(all(feature = "qualcomm", not(feature = "intel"), qnn_sdk))]
impl ActivationOps<Self> for NpuBurnBackend {
    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        if let Some(out) = crate::backends::qualcomm::try_relu(&tensor) {
            return out;
        }
        let nd = tensor::npu_to_ndarray(&tensor);
        tensor::ndarray_to_npu(&<Fx as ActivationOps<Fx>>::relu(nd))
    }

    fn softmax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        if dim + 1 == tensor.shape.len() {
            if let Some(out) = crate::backends::qualcomm::try_softmax(&tensor) {
                return out;
            }
        }
        let nd = tensor::npu_to_ndarray(&tensor);
        tensor::ndarray_to_npu(&<Fx as ActivationOps<Fx>>::softmax(nd, dim))
    }
}

#[cfg(not(all(feature = "qualcomm", not(feature = "intel"), qnn_sdk)))]
impl ActivationOps<Self> for NpuBurnBackend {}

// ===========================================================================
// TransactionOps (all methods have defaults)
// ===========================================================================
impl TransactionOps<Self> for NpuBurnBackend {}
