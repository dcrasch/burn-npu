//! `FloatTensorOps` implementations for all platform variants.

use burn_tensor::backend::ExecutionError;
use burn_tensor::ops::*;
use burn_tensor::ops::{BoolTensor, FloatTensor, IntTensor};
use burn_tensor::{
    BoolDType, Distribution, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
};

#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
use super::tensor::*;
use super::{flex_dev, Fx, NpuBurnBackend, NpuBurnDevice};

// ===========================================================================
// FloatTensorOps — apple: all ops go through MLTensor handles
// ===========================================================================
#[cfg(feature = "apple")]
use super::ffi::*;
#[cfg(feature = "apple")]
use burn_flex::FlexTensor;
#[cfg(feature = "apple")]
use burn_tensor::{f16, DType};

#[cfg(feature = "apple")]
impl FloatTensorOps<Self> for NpuBurnBackend {
    fn float_from_data(data: TensorData, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        // fp16 is the ANE's native format, so f16 input stays f16 rather than
        // being widened to f32 on the way in.
        if data.dtype == DType::F16 {
            let values: Vec<f16> = data.to_vec().unwrap();
            return f16_to_npu(&values, &data.shape.to_vec());
        }
        let floats: Vec<f32> = data.to_vec().unwrap();
        let shape: Vec<i32> = data.shape.iter().map(|&d| d as i32).collect();
        NpuFloatTensor {
            handle: unsafe {
                npu_create_tensor(
                    shape.as_ptr(),
                    shape.len() as i32,
                    floats.as_ptr(),
                    floats.len() as i32,
                )
            },
        }
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NpuBurnDevice,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        // Generate random data on CPU, then send to NPU
        let nd_tensor =
            <Fx as FloatTensorOps<Fx>>::float_random(shape, distribution, &flex_dev(), dtype);
        ndarray_to_npu(&nd_tensor)
    }

    fn float_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        let t = NpuFloatTensor {
            handle: unsafe { npu_zeros(s.as_ptr(), s.len() as i32) },
        };
        Self::float_cast(t, dtype)
    }

    fn float_ones(shape: Shape, _device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        let t = NpuFloatTensor {
            handle: unsafe { npu_ones(s.as_ptr(), s.len() as i32) },
        };
        Self::float_cast(t, dtype)
    }

    fn float_full(
        shape: Shape,
        fill_value: Scalar,
        _device: &NpuBurnDevice,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        let t = NpuFloatTensor {
            handle: unsafe { npu_full(s.as_ptr(), s.len() as i32, fill_value.elem::<f32>()) },
        };
        Self::float_cast(t, dtype)
    }

    fn float_device(_tensor: &FloatTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }

    fn float_to_device(tensor: FloatTensor<Self>, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        tensor
    }

    fn float_empty(shape: Shape, device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        Self::float_zeros(shape, device, dtype)
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        let shape = burn_tensor::TensorMetadata::shape(&tensor);
        if burn_tensor::TensorMetadata::dtype(&tensor) == DType::F16 {
            let (values, _) = read_f16(tensor.handle);
            // tensor drops here, freeing the MLTensor handle
            return Ok(TensorData::new(values, shape));
        }
        let total: usize = shape.num_elements();
        let mut data = vec![0.0f32; total];
        unsafe { npu_get_data(tensor.handle, data.as_mut_ptr(), total as i32) };
        Ok(TensorData::new(data, shape))
    }

    // ── Matmul ──────────────────────────────────────────────────────────

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_matmul(lhs.handle, rhs.handle) },
        }
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        // No FFI for cross product — round-trip through Flex
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_cross(nd_lhs, nd_rhs, dim);
        ndarray_to_npu(&result)
    }

    fn float_into_int(tensor: FloatTensor<Self>, _out_dtype: IntDType) -> IntTensor<Self> {
        let (data, shape) = read_f32(tensor.handle);
        // IntElem is i32 to match burn-flex.
        let int_data: Vec<i32> = data.iter().map(|&v| v as i32).collect();
        FlexTensor::from_data(TensorData::new(int_data, shape))
    }

    // ── Arithmetic ──────────────────────────────────────────────────────

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_add(lhs.handle, rhs.handle) },
        }
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_add_scalar(lhs.handle, rhs.elem::<f32>()) },
        }
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sub(lhs.handle, rhs.handle) },
        }
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sub_scalar(lhs.handle, rhs.elem::<f32>()) },
        }
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_mul(lhs.handle, rhs.handle) },
        }
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_mul_scalar(lhs.handle, rhs.elem::<f32>()) },
        }
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_div(lhs.handle, rhs.handle) },
        }
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_div_scalar(lhs.handle, rhs.elem::<f32>()) },
        }
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        // remainder = lhs - (lhs / rhs).floor() * rhs
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_remainder(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_remainder_scalar(nd_lhs, rhs);
        ndarray_to_npu(&result)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // recip = 1.0 / tensor
        NpuFloatTensor {
            handle: unsafe {
                let one = npu_scalar_tensor(1.0);
                let r = npu_div(one, tensor.handle);
                npu_free_tensor(one);
                r
            },
        }
    }

    // ── Shape / layout ──────────────────────────────────────────────────

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_transpose(tensor.handle, dim1 as i32, dim2 as i32) },
        }
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let perm: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
        NpuFloatTensor {
            handle: unsafe { npu_permute(tensor.handle, perm.as_ptr(), perm.len() as i32) },
        }
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        // No direct FFI — round-trip
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_flip(nd, axes);
        ndarray_to_npu(&result)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        NpuFloatTensor {
            handle: unsafe { npu_reshape(tensor.handle, s.as_ptr(), s.len() as i32) },
        }
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        NpuFloatTensor {
            handle: unsafe { npu_expand(tensor.handle, s.as_ptr(), s.len() as i32) },
        }
    }

    // ── Gather / scatter / select ───────────────────────────────────────

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let idx_data = extract_i64(&indices);
        let idx_i32: Vec<i32> = idx_data.iter().map(|&v| v as i32).collect();
        let idx_len = idx_data.len();
        let idx_shape: Vec<i32> = vec![idx_len as i32];
        unsafe {
            let idx_handle = npu_create_int_tensor(
                idx_shape.as_ptr(),
                idx_shape.len() as i32,
                idx_i32.as_ptr(),
                idx_i32.len() as i32,
            );
            let result = npu_gather(tensor.handle, dim as i32, idx_handle);
            npu_free_tensor(idx_handle);
            NpuFloatTensor { handle: result }
        }
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result =
            <Fx as FloatTensorOps<Fx>>::float_scatter_add(dim, nd_tensor, indices, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        // Native NPU gather — no readback of weight tensor
        let idx_data = extract_i64(&indices);
        let idx_i32: Vec<i32> = idx_data.iter().map(|&v| v as i32).collect();
        let idx_len = idx_data.len();
        let idx_shape: Vec<i32> = vec![idx_len as i32];
        unsafe {
            let idx_handle = npu_create_int_tensor(
                idx_shape.as_ptr(),
                idx_shape.len() as i32,
                idx_i32.as_ptr(),
                idx_i32.len() as i32,
            );
            let result = npu_gather(tensor.handle, dim as i32, idx_handle);
            npu_free_tensor(idx_handle);
            NpuFloatTensor { handle: result }
        }
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result =
            <Fx as FloatTensorOps<Fx>>::float_select_add(nd_tensor, dim, indices, nd_value);
        ndarray_to_npu(&result)
    }

    // ── Slice ───────────────────────────────────────────────────────────

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        // Fast path: simple contiguous ranges (step=1, no negative indices)
        // This covers narrow() which is the common case
        let ndim = slices.len();
        if ndim <= 3 {
            let all_simple = slices.iter().all(|s| s.step == 1 && s.start >= 0);
            if all_simple {
                let mut ranges = Vec::with_capacity(ndim * 2);
                let shape = {
                    let mut buf = [0i32; 8];
                    let n = unsafe { npu_get_shape(tensor.handle, buf.as_mut_ptr(), 8) } as usize;
                    buf[..n].iter().map(|&d| d as usize).collect::<Vec<_>>()
                };
                for (i, s) in slices.iter().enumerate() {
                    let start = s.start as i32;
                    let end = s.end.map(|e| e as i32).unwrap_or(shape[i] as i32);
                    ranges.push(start);
                    ranges.push(end);
                }
                let result = unsafe { npu_slice(tensor.handle, ranges.as_ptr(), ndim as i32) };
                if result >= 0 {
                    return NpuFloatTensor { handle: result };
                }
            }
        }
        // Fallback for complex slices
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_slice(nd, slices);
        ndarray_to_npu(&result)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // No direct FFI for slice_assign — round-trip
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Fx as FloatTensorOps<Fx>>::float_slice_assign(nd_tensor, slices, nd_value);
        ndarray_to_npu(&result)
    }

    // ── Mask ────────────────────────────────────────────────────────────

    // Masking stays on the NPU. Uploading the mask keeps the value tensor's
    // MLTensor graph lazy; round-tripping it through the CPU would force the
    // whole pending computation to materialise, which on an attention mask is
    // the hot path.
    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let mask = bool_mask_to_npu(mask);
        NpuFloatTensor {
            handle: unsafe { npu_mask_where(tensor.handle, mask.handle, value.handle) },
        }
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> FloatTensor<Self> {
        let mask = bool_mask_to_npu(mask);
        NpuFloatTensor {
            handle: unsafe { npu_mask_fill(tensor.handle, mask.handle, value.elem::<f32>()) },
        }
    }

    // ── Comparison (return BoolTensor = FlexTensor<bool>) ────────────

    fn float_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        _out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let h = unsafe { npu_equal(lhs.handle, rhs.handle) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        _out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs.elem::<f32>()) };
        let h = unsafe { npu_equal(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_greater(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        _out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let h = unsafe { npu_greater(lhs.handle, rhs.handle) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_greater_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        _out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs.elem::<f32>()) };
        let h = unsafe { npu_greater(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_greater_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        _out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        // greater_equal = NOT less
        let h = unsafe { npu_less(lhs.handle, rhs.handle) };
        float_handle_to_inverted_bool(h)
    }

    fn float_greater_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        _out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs.elem::<f32>()) };
        let h = unsafe { npu_less(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        float_handle_to_inverted_bool(h)
    }

    fn float_lower(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        _out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let h = unsafe { npu_less(lhs.handle, rhs.handle) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_lower_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        _out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs.elem::<f32>()) };
        let h = unsafe { npu_less(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_lower_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        _out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        // lower_equal = NOT greater
        let h = unsafe { npu_greater(lhs.handle, rhs.handle) };
        float_handle_to_inverted_bool(h)
    }

    fn float_lower_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        _out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs.elem::<f32>()) };
        let h = unsafe { npu_greater(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        float_handle_to_inverted_bool(h)
    }

    // ── Reductions ──────────────────────────────────────────────────────

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sum(tensor.handle) },
        }
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sum_dim(tensor.handle, dim as i32) },
        }
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_mean_all(tensor.handle) },
        }
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_mean(tensor.handle, dim as i32) },
        }
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_prod(nd);
        ndarray_to_npu(&result)
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_prod_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_cumsum(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_cumprod(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_cummin(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_cummax(nd, dim);
        ndarray_to_npu(&result)
    }

    // ── Argmax / Argmin (return IntTensor = FlexTensor) ──────────────

    fn float_argmax(
        tensor: FloatTensor<Self>,
        dim: usize,
        _out_dtype: IntDType,
    ) -> IntTensor<Self> {
        let h = unsafe { npu_argmax(tensor.handle, dim as i32) };
        int_handle_to_ndarray(h)
    }

    fn float_argmin(
        tensor: FloatTensor<Self>,
        dim: usize,
        _out_dtype: IntDType,
    ) -> IntTensor<Self> {
        let h = unsafe { npu_argmin(tensor.handle, dim as i32) };
        int_handle_to_ndarray(h)
    }
    fn float_argtopk(
        tensor: FloatTensor<Self>,
        dim: usize,
        k: usize,
        out_dtype: IntDType,
    ) -> IntTensor<Self> {
        // No MLTensor primitive for top-k; round-trip through the CPU delegate.
        let nd = npu_to_ndarray(&tensor);
        <Fx as FloatTensorOps<Fx>>::float_argtopk(nd, dim, k, out_dtype)
    }

    // ── Max / Min ───────────────────────────────────────────────────────

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_max(tensor.handle) },
        }
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_max_dim(tensor.handle, dim as i32) },
        }
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_min(tensor.handle) },
        }
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_min_dim(tensor.handle, dim as i32) },
        }
    }

    // ── Unary math ──────────────────────────────────────────────────────

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_exp(tensor.handle) },
        }
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_log(tensor.handle) },
        }
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // log1p(x) = log(1 + x)
        let one = unsafe { npu_scalar_tensor(1.0) };
        let sum = unsafe { npu_add(tensor.handle, one) };
        let result = unsafe { npu_log(sum) };
        unsafe {
            npu_free_tensor(one);
            npu_free_tensor(sum);
        }
        NpuFloatTensor { handle: result }
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_pow(lhs.handle, rhs.handle) },
        }
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_pow_scalar(tensor.handle, value.elem::<f32>()) },
        }
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sqrt(tensor.handle) },
        }
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_abs(tensor.handle) },
        }
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_cos(tensor.handle) },
        }
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sin(tensor.handle) },
        }
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_tanh(tensor.handle) },
        }
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_erf(tensor.handle) },
        }
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_floor(tensor.handle) },
        }
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_ceil(tensor.handle) },
        }
    }

    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_neg(tensor.handle) },
        }
    }

    // Trig ops without direct FFI — round-trip through Flex
    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_tan(nd);
        ndarray_to_npu(&result)
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_cosh(nd);
        ndarray_to_npu(&result)
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_sinh(nd);
        ndarray_to_npu(&result)
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_acos(nd);
        ndarray_to_npu(&result)
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_acosh(nd);
        ndarray_to_npu(&result)
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_asin(nd);
        ndarray_to_npu(&result)
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_asinh(nd);
        ndarray_to_npu(&result)
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_atan(nd);
        ndarray_to_npu(&result)
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_atanh(nd);
        ndarray_to_npu(&result)
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_atan2(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_round(nd);
        ndarray_to_npu(&result)
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_trunc(nd);
        ndarray_to_npu(&result)
    }

    // ── Clamp ───────────────────────────────────────────────────────────

    fn float_clamp_min(tensor: FloatTensor<Self>, min: Scalar) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_clamp_min(tensor.handle, min.elem::<f32>()) },
        }
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: Scalar) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_clamp_max(tensor.handle, max.elem::<f32>()) },
        }
    }

    fn float_clamp(tensor: FloatTensor<Self>, min: Scalar, max: Scalar) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_clamp(tensor.handle, min.elem::<f32>(), max.elem::<f32>()) },
        }
    }

    // ── Cat ─────────────────────────────────────────────────────────────

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        let handles: Vec<i32> = tensors.iter().map(|t| t.handle).collect();
        NpuFloatTensor {
            handle: unsafe { npu_cat(handles.as_ptr(), handles.len() as i32, dim as i32) },
        }
    }

    // ── Sign ────────────────────────────────────────────────────────────

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_sign(nd);
        ndarray_to_npu(&result)
    }

    // ── Cast ────────────────────────────────────────────────────────────

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let code = match DType::from(dtype) {
            DType::F16 => DTYPE_F16,
            _ => DTYPE_F32,
        };
        NpuFloatTensor {
            handle: unsafe { npu_cast_float(tensor.handle, code) },
        }
    }

    // ── Grid sample ─────────────────────────────────────────────────────

    fn float_grid_sample_2d(
        tensor: FloatTensor<Self>,
        grid: FloatTensor<Self>,
        options: GridSampleOptions,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_grid = npu_to_ndarray(&grid);
        let result = <Fx as FloatTensorOps<Fx>>::float_grid_sample_2d(nd_tensor, nd_grid, options);
        ndarray_to_npu(&result)
    }

    // ── Unfold ──────────────────────────────────────────────────────────

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_unfold(nd, dim, size, step);
        ndarray_to_npu(&result)
    }
}

// ===========================================================================
// FloatTensorOps — no feature: full Flex delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl FloatTensorOps<Self> for NpuBurnBackend {
    fn float_from_data(data: TensorData, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_from_data(data, &flex_dev())
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NpuBurnDevice,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_random(shape, distribution, &flex_dev(), dtype)
    }

    fn float_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_zeros(shape, &flex_dev(), dtype)
    }

    fn float_ones(shape: Shape, _device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_ones(shape, &flex_dev(), dtype)
    }

    fn float_full(
        shape: Shape,
        fill_value: Scalar,
        _device: &NpuBurnDevice,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_full(shape, fill_value, &flex_dev(), dtype)
    }

    fn float_device(_tensor: &FloatTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }
    fn float_to_device(tensor: FloatTensor<Self>, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        tensor
    }

    fn float_empty(shape: Shape, _device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_empty(shape, &flex_dev(), dtype)
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Fx as FloatTensorOps<Fx>>::float_into_data(tensor).await
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_matmul(lhs, rhs)
    }
    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_cross(lhs, rhs, dim)
    }
    fn float_into_int(tensor: FloatTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_into_int(tensor, out_dtype)
    }
    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_add(lhs, rhs)
    }
    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_add_scalar(lhs, rhs)
    }
    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_sub(lhs, rhs)
    }
    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_sub_scalar(lhs, rhs)
    }
    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_mul(lhs, rhs)
    }
    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_mul_scalar(lhs, rhs)
    }
    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_div(lhs, rhs)
    }
    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_div_scalar(lhs, rhs)
    }
    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_remainder(lhs, rhs)
    }
    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_remainder_scalar(lhs, rhs)
    }
    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_recip(tensor)
    }
    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_swap_dims(tensor, dim1, dim2)
    }
    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_permute(tensor, axes)
    }
    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_flip(tensor, axes)
    }
    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_reshape(tensor, shape)
    }
    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_expand(tensor, shape)
    }
    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_gather(dim, tensor, indices)
    }
    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_scatter_add(dim, tensor, indices, value)
    }
    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_select(tensor, dim, indices)
    }
    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_select_add(tensor, dim, indices, value)
    }
    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_slice(tensor, slices)
    }
    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_slice_assign(tensor, slices, value)
    }
    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_mask_where(tensor, mask, value)
    }
    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_mask_fill(tensor, mask, value)
    }
    fn float_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_equal(lhs, rhs, out_dtype)
    }
    fn float_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_equal_elem(lhs, rhs, out_dtype)
    }
    fn float_greater(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_greater(lhs, rhs, out_dtype)
    }
    fn float_greater_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_greater_elem(lhs, rhs, out_dtype)
    }
    fn float_greater_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_greater_equal(lhs, rhs, out_dtype)
    }
    fn float_greater_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_greater_equal_elem(lhs, rhs, out_dtype)
    }
    fn float_lower(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_lower(lhs, rhs, out_dtype)
    }
    fn float_lower_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_lower_elem(lhs, rhs, out_dtype)
    }
    fn float_lower_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_lower_equal(lhs, rhs, out_dtype)
    }
    fn float_lower_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_lower_equal_elem(lhs, rhs, out_dtype)
    }
    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_sum(tensor)
    }
    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_sum_dim(tensor, dim)
    }
    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_mean(tensor)
    }
    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_mean_dim(tensor, dim)
    }
    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_prod(tensor)
    }
    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_prod_dim(tensor, dim)
    }
    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_cumsum(tensor, dim)
    }
    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_cumprod(tensor, dim)
    }
    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_cummin(tensor, dim)
    }
    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_cummax(tensor, dim)
    }
    fn float_argmax(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_argmax(tensor, dim, out_dtype)
    }
    fn float_argmin(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_argmin(tensor, dim, out_dtype)
    }
    fn float_argtopk(
        tensor: FloatTensor<Self>,
        dim: usize,
        k: usize,
        out_dtype: IntDType,
    ) -> IntTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_argtopk(tensor, dim, k, out_dtype)
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_exp(tensor)
    }
    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_log(tensor)
    }
    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_log1p(tensor)
    }
    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_powf(lhs, rhs)
    }
    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_powf_scalar_impl(tensor, value)
    }
    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_sqrt(tensor)
    }
    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_abs(tensor)
    }
    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_cos(tensor)
    }
    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_sin(tensor)
    }
    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_tan(tensor)
    }
    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_cosh(tensor)
    }
    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_sinh(tensor)
    }
    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_tanh(tensor)
    }
    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_acos(tensor)
    }
    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_acosh(tensor)
    }
    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_asin(tensor)
    }
    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_asinh(tensor)
    }
    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_atan(tensor)
    }
    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_atanh(tensor)
    }
    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_atan2(lhs, rhs)
    }
    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_round(tensor)
    }
    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_floor(tensor)
    }
    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_ceil(tensor)
    }
    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_trunc(tensor)
    }
    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_erf(tensor)
    }
    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_cat(tensors, dim)
    }
    fn float_clamp_min(tensor: FloatTensor<Self>, min: Scalar) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_clamp_min(tensor, min)
    }
    fn float_clamp_max(tensor: FloatTensor<Self>, max: Scalar) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_clamp_max(tensor, max)
    }
    fn float_clamp(tensor: FloatTensor<Self>, min: Scalar, max: Scalar) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_clamp(tensor, min, max)
    }
    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_neg(tensor)
    }
    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_sign(tensor)
    }
    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_cast(tensor, dtype)
    }
    fn float_grid_sample_2d(
        tensor: FloatTensor<Self>,
        grid: FloatTensor<Self>,
        options: GridSampleOptions,
    ) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_grid_sample_2d(tensor, grid, options)
    }
    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_unfold(tensor, dim, size, step)
    }
    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_max(tensor)
    }
    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_max_dim(tensor, dim)
    }
    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_min(tensor)
    }
    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Fx as FloatTensorOps<Fx>>::float_min_dim(tensor, dim)
    }
}

// ===========================================================================
// FloatTensorOps — intel/qualcomm: Vec<f32> tensor, Flex delegation
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl FloatTensorOps<Self> for NpuBurnBackend {
    fn float_from_data(data: TensorData, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        let floats: Vec<f32> = data.to_vec().unwrap();
        let shape: Vec<usize> = data.shape.to_vec();
        NpuFloatTensor::new(floats, shape)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NpuBurnDevice,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        let nd_tensor =
            <Fx as FloatTensorOps<Fx>>::float_random(shape, distribution, &flex_dev(), dtype);
        ndarray_to_npu(&nd_tensor)
    }

    fn float_zeros(shape: Shape, _device: &NpuBurnDevice, _dtype: FloatDType) -> FloatTensor<Self> {
        NpuFloatTensor::zeros(shape.to_vec())
    }

    fn float_ones(shape: Shape, _device: &NpuBurnDevice, _dtype: FloatDType) -> FloatTensor<Self> {
        NpuFloatTensor::ones(shape.to_vec())
    }

    fn float_full(
        shape: Shape,
        fill_value: Scalar,
        _device: &NpuBurnDevice,
        _dtype: FloatDType,
    ) -> FloatTensor<Self> {
        NpuFloatTensor::full(shape.to_vec(), fill_value.elem::<f32>())
    }

    fn float_device(_tensor: &FloatTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }

    fn float_to_device(tensor: FloatTensor<Self>, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        tensor
    }

    fn float_empty(shape: Shape, device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        Self::float_zeros(shape, device, dtype)
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        let shape = burn_tensor::TensorMetadata::shape(&tensor);
        Ok(TensorData::new(tensor.data, shape))
    }

    // ── Matmul ──────────────────────────────────────────────────────────

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        // Intel: try OpenVINO NPU for large matmuls, else CPU.
        // Takes precedence if both features are somehow enabled at once.
        #[cfg(feature = "intel")]
        {
            crate::backends::intel::openvino_matmul(&lhs, &rhs)
                .unwrap_or_else(|_| crate::backends::intel::cpu_matmul(&lhs, &rhs))
        }

        // Qualcomm: Hexagon NPU via QNN when available, else CPU.
        #[cfg(all(feature = "qualcomm", not(feature = "intel")))]
        {
            crate::backends::qualcomm::matmul(&lhs, &rhs)
        }
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_cross(nd_lhs, nd_rhs, dim);
        ndarray_to_npu(&result)
    }

    fn float_into_int(tensor: FloatTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        <Fx as FloatTensorOps<Fx>>::float_into_int(nd, out_dtype)
    }

    // ── Arithmetic ──────────────────────────────────────────────────────

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_add(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_add_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_sub(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_sub_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_mul(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_mul_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_div(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_div_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_remainder(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_remainder_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_recip(nd);
        ndarray_to_npu(&result)
    }

    // ── Shape / layout ──────────────────────────────────────────────────

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_swap_dims(nd, dim1, dim2);
        ndarray_to_npu(&result)
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_permute(nd, axes);
        ndarray_to_npu(&result)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_flip(nd, axes);
        ndarray_to_npu(&result)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_reshape(nd, shape);
        ndarray_to_npu(&result)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_expand(nd, shape);
        ndarray_to_npu(&result)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_gather(dim, nd, indices);
        ndarray_to_npu(&result)
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result =
            <Fx as FloatTensorOps<Fx>>::float_scatter_add(dim, nd_tensor, indices, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_select(nd, dim, indices);
        ndarray_to_npu(&result)
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result =
            <Fx as FloatTensorOps<Fx>>::float_select_add(nd_tensor, dim, indices, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_slice(nd, slices);
        ndarray_to_npu(&result)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Fx as FloatTensorOps<Fx>>::float_slice_assign(nd_tensor, slices, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Fx as FloatTensorOps<Fx>>::float_mask_where(nd_tensor, mask, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_mask_fill(nd_tensor, mask, value);
        ndarray_to_npu(&result)
    }

    fn float_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Fx as FloatTensorOps<Fx>>::float_equal(nd_lhs, nd_rhs, out_dtype)
    }

    fn float_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Fx as FloatTensorOps<Fx>>::float_equal_elem(nd, rhs, out_dtype)
    }

    fn float_greater(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Fx as FloatTensorOps<Fx>>::float_greater(nd_lhs, nd_rhs, out_dtype)
    }

    fn float_greater_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Fx as FloatTensorOps<Fx>>::float_greater_elem(nd, rhs, out_dtype)
    }

    fn float_greater_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Fx as FloatTensorOps<Fx>>::float_greater_equal(nd_lhs, nd_rhs, out_dtype)
    }

    fn float_greater_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Fx as FloatTensorOps<Fx>>::float_greater_equal_elem(nd, rhs, out_dtype)
    }

    fn float_lower(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Fx as FloatTensorOps<Fx>>::float_lower(nd_lhs, nd_rhs, out_dtype)
    }

    fn float_lower_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Fx as FloatTensorOps<Fx>>::float_lower_elem(nd, rhs, out_dtype)
    }

    fn float_lower_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Fx as FloatTensorOps<Fx>>::float_lower_equal(nd_lhs, nd_rhs, out_dtype)
    }

    fn float_lower_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Fx as FloatTensorOps<Fx>>::float_lower_equal_elem(nd, rhs, out_dtype)
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_sum(nd);
        ndarray_to_npu(&result)
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_sum_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_mean(nd);
        ndarray_to_npu(&result)
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_mean_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_prod(nd);
        ndarray_to_npu(&result)
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_prod_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_cumsum(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_cumprod(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_cummin(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_cummax(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        <Fx as FloatTensorOps<Fx>>::float_argmax(nd, dim, out_dtype)
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        <Fx as FloatTensorOps<Fx>>::float_argmin(nd, dim, out_dtype)
    }
    fn float_argtopk(
        tensor: FloatTensor<Self>,
        dim: usize,
        k: usize,
        out_dtype: IntDType,
    ) -> IntTensor<Self> {
        // No MLTensor primitive for top-k; round-trip through the CPU delegate.
        let nd = npu_to_ndarray(&tensor);
        <Fx as FloatTensorOps<Fx>>::float_argtopk(nd, dim, k, out_dtype)
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_max(nd);
        ndarray_to_npu(&result)
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_max_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_min(nd);
        ndarray_to_npu(&result)
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_min_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_exp(nd);
        ndarray_to_npu(&result)
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_log(nd);
        ndarray_to_npu(&result)
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_log1p(nd);
        ndarray_to_npu(&result)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_powf(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_powf_scalar_impl(nd, value);
        ndarray_to_npu(&result)
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_sqrt(nd);
        ndarray_to_npu(&result)
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_abs(nd);
        ndarray_to_npu(&result)
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_cos(nd);
        ndarray_to_npu(&result)
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_sin(nd);
        ndarray_to_npu(&result)
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_tanh(nd);
        ndarray_to_npu(&result)
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_erf(nd);
        ndarray_to_npu(&result)
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_floor(nd);
        ndarray_to_npu(&result)
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_ceil(nd);
        ndarray_to_npu(&result)
    }

    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_neg(nd);
        ndarray_to_npu(&result)
    }

    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_tan(nd);
        ndarray_to_npu(&result)
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_cosh(nd);
        ndarray_to_npu(&result)
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_sinh(nd);
        ndarray_to_npu(&result)
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_acos(nd);
        ndarray_to_npu(&result)
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_acosh(nd);
        ndarray_to_npu(&result)
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_asin(nd);
        ndarray_to_npu(&result)
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_asinh(nd);
        ndarray_to_npu(&result)
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_atan(nd);
        ndarray_to_npu(&result)
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_atanh(nd);
        ndarray_to_npu(&result)
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Fx as FloatTensorOps<Fx>>::float_atan2(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_round(nd);
        ndarray_to_npu(&result)
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_trunc(nd);
        ndarray_to_npu(&result)
    }

    fn float_clamp_min(tensor: FloatTensor<Self>, min: Scalar) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_clamp_min(nd, min);
        ndarray_to_npu(&result)
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: Scalar) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_clamp_max(nd, max);
        ndarray_to_npu(&result)
    }

    fn float_clamp(tensor: FloatTensor<Self>, min: Scalar, max: Scalar) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_clamp(nd, min, max);
        ndarray_to_npu(&result)
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        #[cfg(all(feature = "qualcomm", not(feature = "intel"), qnn_sdk))]
        {
            let refs: Vec<&NpuFloatTensor> = tensors.iter().collect();
            if let Some(out) = crate::backends::qualcomm::try_concat(&refs, dim) {
                return out;
            }
        }
        let nd_tensors: Vec<_> = tensors.iter().map(npu_to_ndarray).collect();
        let result = <Fx as FloatTensorOps<Fx>>::float_cat(nd_tensors, dim);
        ndarray_to_npu(&result)
    }

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_sign(nd);
        ndarray_to_npu(&result)
    }

    fn float_cast(tensor: FloatTensor<Self>, _dtype: FloatDType) -> FloatTensor<Self> {
        // Only f32 supported; casting is a no-op.
        tensor
    }

    fn float_grid_sample_2d(
        tensor: FloatTensor<Self>,
        grid: FloatTensor<Self>,
        options: GridSampleOptions,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_grid = npu_to_ndarray(&grid);
        let result = <Fx as FloatTensorOps<Fx>>::float_grid_sample_2d(nd_tensor, nd_grid, options);
        ndarray_to_npu(&result)
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Fx as FloatTensorOps<Fx>>::float_unfold(nd, dim, size, step);
        ndarray_to_npu(&result)
    }
}
