//! Qualcomm Hexagon NPU backend via the QNN (AI Engine Direct) SDK.
//!
//! `QnnFloatTensor` stores data as `Vec<f32>` with shape metadata. Matmul is
//! dispatched to the Hexagon Tensor Processor when a QAIRT SDK was present at
//! build time and the HTP backend library can be opened at runtime; everything
//! else runs on CPU or delegates to burn-flex.
//!
//! # Build requirements
//!
//! NPU dispatch is compiled in only when `QNN_SDK_ROOT` points at a Qualcomm AI
//! Runtime (QAIRT) SDK during the build -- see [`crate`]'s `build.rs`. Bindings
//! are generated from the SDK's own headers rather than hand-written, because
//! the QNN ABI uses versioned tagged unions that are unsafe to transcribe by
//! hand. Without the SDK the feature still builds and runs everything on CPU.
//!
//! # Status
//!
//! The NPU path is **implemented but awaiting hardware validation** -- see
//! [`qnn`] for what specifically still needs checking on a device.

use burn_tensor::{DType, Shape};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(qnn_sdk)]
pub mod qnn;

/// One static parameter of a QNN op, using the names from `QnnOpDef.h`.
pub enum OpParam {
    /// Scalar `uint32` such as `group` or `axis`.
    U32(&'static str, u32),
    /// Scalar `float32` such as `beta`.
    F32(&'static str, f32),
    /// Scalar `bool8` such as `count_pad_for_edges`.
    Bool(&'static str, bool),
    /// `uint32` tensor parameter such as `stride` or `pad_amount`: dims, values.
    U32Tensor(&'static str, Vec<u32>, Vec<u32>),
}

/// A tensor in a fused graph: its dims and, for weights, its data.
pub struct FusedTensor {
    pub dims: Vec<u32>,
    /// `Some` for static weights and biases, `None` for the graph input and
    /// all intermediates.
    pub data: Option<Vec<f32>>,
}

/// One op in a fused graph. Tensors are indexes into [`FusedGraph::tensors`].
pub struct FusedOp {
    pub op_type: &'static str,
    pub inputs: Vec<usize>,
    pub output: usize,
    pub params: Vec<OpParam>,
}

/// A whole network described as tensors plus ops, built into one QNN graph.
pub struct FusedGraph {
    pub tensors: Vec<FusedTensor>,
    pub ops: Vec<FusedOp>,
    pub input: usize,
    pub output: usize,
}

/// Number of matmuls that have actually executed on the Hexagon NPU.
///
/// The CPU fallback is intentionally silent, so this is the only way for a
/// caller to tell a QNN dispatch from a fallback.
static NPU_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// How many matmuls have run on the Hexagon NPU in this process.
pub fn npu_dispatch_count() -> u64 {
    NPU_DISPATCHES.load(Ordering::Relaxed)
}

/// Total time spent building QNN graphs in this process, in nanoseconds.
///
/// Graph building happens on the first call for each distinct op signature; it
/// is zero once the graphs are cached.
static NPU_BUILD_NANOS: AtomicU64 = AtomicU64::new(0);
/// Total time spent executing QNN graphs in this process, in nanoseconds.
static NPU_EXEC_NANOS: AtomicU64 = AtomicU64::new(0);

/// Nanoseconds spent building QNN graphs so far.
pub fn npu_build_nanos() -> u64 {
    NPU_BUILD_NANOS.load(Ordering::Relaxed)
}

/// Nanoseconds spent executing QNN graphs so far.
pub fn npu_exec_nanos() -> u64 {
    NPU_EXEC_NANOS.load(Ordering::Relaxed)
}

#[cfg(qnn_sdk)]
pub(crate) fn record_build_time(nanos: u64) {
    NPU_BUILD_NANOS.fetch_add(nanos, Ordering::Relaxed);
}

#[cfg(qnn_sdk)]
pub(crate) fn record_exec_time(nanos: u64) {
    NPU_EXEC_NANOS.fetch_add(nanos, Ordering::Relaxed);
}

/// When set, every op takes the CPU path even if the NPU is available. Used
/// to benchmark CPU against NPU in one process.
static FORCE_CPU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Route all ops to the CPU (`true`) or allow the NPU again (`false`).
pub fn set_force_cpu(force: bool) {
    FORCE_CPU.store(force, Ordering::Relaxed);
}

/// Whether the NPU is currently allowed.
pub fn npu_enabled() -> bool {
    cfg!(qnn_sdk) && !FORCE_CPU.load(Ordering::Relaxed)
}

/// Whether this build was compiled with QNN bindings (`QNN_SDK_ROOT` set).
pub const fn qnn_compiled() -> bool {
    cfg!(qnn_sdk)
}

/// Returned when Hexagon NPU dispatch is unavailable and the caller should fall
/// back to CPU.
///
/// Expected rather than exceptional: it is the normal result on any machine
/// built without the QNN SDK, or without a Snapdragon device present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QnnUnavailable;

impl std::fmt::Display for QnnUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("QNN runtime or Hexagon device unavailable; falling back to CPU")
    }
}

impl std::error::Error for QnnUnavailable {}

/// Attempt Hexagon NPU dispatch, falling back to CPU.
///
/// This is the single entry point the burn backend calls for matmul.
pub fn matmul(lhs: &QnnFloatTensor, rhs: &QnnFloatTensor) -> QnnFloatTensor {
    #[cfg(qnn_sdk)]
    {
        if let Ok(result) = qnn::qnn_matmul(lhs, rhs) {
            NPU_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            return result;
        }
    }
    cpu_matmul(lhs, rhs)
}

/// Try a conv2d on the NPU. `None` means the caller should use the CPU path.
#[cfg(qnn_sdk)]
pub fn try_conv2d(
    x: &QnnFloatTensor,
    weight: &QnnFloatTensor,
    bias: Option<&QnnFloatTensor>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Option<QnnFloatTensor> {
    if !npu_enabled() {
        return None;
    }
    let result = qnn::qnn_conv2d(x, weight, bias, stride, padding, dilation, groups).ok()?;
    NPU_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    Some(result)
}

/// Try a 2D pooling on the NPU. `None` means the caller should use the CPU path.
#[cfg(qnn_sdk)]
pub fn try_pool2d(
    x: &QnnFloatTensor,
    max: bool,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> Option<QnnFloatTensor> {
    if !npu_enabled() {
        return None;
    }
    let result = qnn::qnn_pool2d(x, max, kernel, stride, padding, count_include_pad).ok()?;
    NPU_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    Some(result)
}

/// Try a ReLU on the NPU. `None` means the caller should use the CPU path.
#[cfg(qnn_sdk)]
pub fn try_relu(x: &QnnFloatTensor) -> Option<QnnFloatTensor> {
    if !npu_enabled() {
        return None;
    }
    let result = qnn::qnn_relu(x).ok()?;
    NPU_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    Some(result)
}

/// Run a whole network as one fused NPU graph. `key` names the graph and must
/// be the same for every call with the same `graph`. Counts one dispatch per
/// op in the graph. `None` means CPU.
pub fn run_fused(key: &'static str, graph: &'static FusedGraph, input: &[f32]) -> Option<Vec<f32>> {
    #[cfg(qnn_sdk)]
    {
        if npu_enabled() {
            if let Ok(out) = qnn::qnn_fused(key, graph, input) {
                NPU_DISPATCHES.fetch_add(graph.ops.len() as u64, Ordering::Relaxed);
                return Some(out);
            }
        }
    }
    let _ = (key, graph, input);
    None
}

/// Try a softmax over the last axis on the NPU. `None` means CPU.
#[cfg(qnn_sdk)]
pub fn try_softmax(x: &QnnFloatTensor) -> Option<QnnFloatTensor> {
    if !npu_enabled() {
        return None;
    }
    let result = qnn::qnn_softmax(x).ok()?;
    NPU_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    Some(result)
}

/// Try a concat on the NPU. `None` means the caller should use the CPU path.
#[cfg(qnn_sdk)]
pub fn try_concat(tensors: &[&QnnFloatTensor], axis: usize) -> Option<QnnFloatTensor> {
    if !npu_enabled() {
        return None;
    }
    let result = qnn::qnn_concat(tensors, axis).ok()?;
    NPU_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    Some(result)
}

// ---------------------------------------------------------------------------
// QnnFloatTensor
// ---------------------------------------------------------------------------

/// A float tensor backed by a `Vec<f32>` with shape metadata.
///
/// When QNN SDK integration is complete, compute-heavy ops like matmul will
/// dispatch to the Hexagon NPU. Currently all ops execute on CPU.
#[derive(Debug, Clone)]
pub struct QnnFloatTensor {
    /// Flat f32 data in row-major order.
    pub data: Vec<f32>,
    /// Shape dimensions.
    pub shape: Vec<usize>,
}

impl QnnFloatTensor {
    /// Create a tensor from flat data and shape.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "QnnFloatTensor::new: data.len() != product(shape)"
        );
        Self { data, shape }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Self {
            data: vec![0.0; total],
            shape,
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Self {
            data: vec![1.0; total],
            shape,
        }
    }

    /// Create a tensor filled with a constant value.
    pub fn full(shape: Vec<usize>, value: f32) -> Self {
        let total: usize = shape.iter().product();
        Self {
            data: vec![value; total],
            shape,
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

impl burn_tensor::TensorMetadata for QnnFloatTensor {
    fn dtype(&self) -> DType {
        DType::F32
    }

    fn shape(&self) -> Shape {
        Shape::from(self.shape.clone())
    }
}

// ---------------------------------------------------------------------------
// CPU matmul
// ---------------------------------------------------------------------------

/// CPU matmul, used whenever NPU dispatch is unavailable.
///
/// Prefer [`matmul`], which tries the Hexagon NPU first.
pub fn cpu_matmul(lhs: &QnnFloatTensor, rhs: &QnnFloatTensor) -> QnnFloatTensor {
    // Support batched matmul: [..., M, K] x [..., K, N] -> [..., M, N]
    let lhs_ndim = lhs.shape.len();
    let rhs_ndim = rhs.shape.len();

    assert!(
        lhs_ndim >= 2 && rhs_ndim >= 2,
        "matmul requires at least 2D tensors"
    );

    let m = lhs.shape[lhs_ndim - 2];
    let k = lhs.shape[lhs_ndim - 1];
    let n = rhs.shape[rhs_ndim - 1];
    assert_eq!(
        rhs.shape[rhs_ndim - 2],
        k,
        "matmul inner dimensions mismatch"
    );

    // Compute batch dimensions.
    let lhs_batch: usize = lhs.shape[..lhs_ndim - 2].iter().product();
    let rhs_batch: usize = rhs.shape[..rhs_ndim - 2].iter().product();
    let batch = lhs_batch.max(rhs_batch);

    let mut out_shape: Vec<usize> = if lhs_ndim >= rhs_ndim {
        lhs.shape[..lhs_ndim - 2].to_vec()
    } else {
        rhs.shape[..rhs_ndim - 2].to_vec()
    };
    out_shape.push(m);
    out_shape.push(n);

    let mut result = vec![0.0f32; batch * m * n];

    for b in 0..batch {
        let lhs_offset = (b % lhs_batch) * m * k;
        let rhs_offset = (b % rhs_batch) * k * n;
        let out_offset = b * m * n;

        for i in 0..m {
            let out_row = &mut result[out_offset + i * n..out_offset + (i + 1) * n];
            for p in 0..k {
                let lhs_val = lhs.data[lhs_offset + i * k + p];
                if lhs_val == 0.0 {
                    continue;
                }
                let rhs_row = &rhs.data[rhs_offset + p * n..rhs_offset + (p + 1) * n];
                for (o, &r) in out_row.iter_mut().zip(rhs_row) {
                    *o += lhs_val * r;
                }
            }
        }
    }

    QnnFloatTensor::new(result, out_shape)
}

// ---------------------------------------------------------------------------
// Conversion helpers for Flex interop
// ---------------------------------------------------------------------------

/// Convert QnnFloatTensor -> FlexTensor (for delegating ops to burn-flex).
pub fn qnn_to_ndarray(tensor: &QnnFloatTensor) -> burn_flex::FlexTensor {
    burn_flex::FlexTensor::from_data(burn_tensor::TensorData::new(
        tensor.data.clone(),
        tensor.shape.clone(),
    ))
}

/// Convert FlexTensor (f32) -> QnnFloatTensor.
pub fn ndarray_to_qnn(tensor: &burn_flex::FlexTensor) -> QnnFloatTensor {
    assert_eq!(
        burn_tensor::TensorMetadata::dtype(tensor),
        DType::F32,
        "ndarray_to_qnn: expected an f32 tensor"
    );
    let contig = tensor.to_contiguous();
    let shape = burn_tensor::TensorMetadata::shape(tensor).to_vec();
    QnnFloatTensor::new(contig.storage::<f32>().to_vec(), shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straightforward triple-loop reference to check the blocked kernel against.
    fn reference(lhs: &QnnFloatTensor, rhs: &QnnFloatTensor) -> Vec<f32> {
        let ln = lhs.shape.len();
        let rn = rhs.shape.len();
        let (m, k, n) = (lhs.shape[ln - 2], lhs.shape[ln - 1], rhs.shape[rn - 1]);
        let batch: usize = lhs.shape[..ln - 2].iter().product::<usize>().max(1);
        let mut out = vec![0.0f32; batch * m * n];
        for b in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..k {
                        sum += lhs.data[b * m * k + i * k + p] * rhs.data[b * k * n + p * n + j];
                    }
                    out[b * m * n + i * n + j] = sum;
                }
            }
        }
        out
    }

    fn seeded(len: usize, seed: u32) -> Vec<f32> {
        // Small LCG; deterministic and dependency-free.
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1 << 24) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn cpu_matmul_2d_matches_reference() {
        let lhs = QnnFloatTensor::new(seeded(6 * 5, 1), vec![6, 5]);
        let rhs = QnnFloatTensor::new(seeded(5 * 4, 2), vec![5, 4]);
        let got = cpu_matmul(&lhs, &rhs);
        assert_eq!(got.shape, vec![6, 4]);
        for (a, b) in got.data.iter().zip(reference(&lhs, &rhs)) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn cpu_matmul_batched_matches_reference() {
        let lhs = QnnFloatTensor::new(seeded(3 * 4 * 7, 3), vec![3, 4, 7]);
        let rhs = QnnFloatTensor::new(seeded(3 * 7 * 2, 4), vec![3, 7, 2]);
        let got = cpu_matmul(&lhs, &rhs);
        assert_eq!(got.shape, vec![3, 4, 2]);
        for (a, b) in got.data.iter().zip(reference(&lhs, &rhs)) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    /// The kernel skips zero multipliers; make sure that shortcut cannot leave
    /// stale accumulator state behind.
    #[test]
    fn cpu_matmul_handles_zero_rows() {
        let lhs = QnnFloatTensor::new(vec![0.0, 0.0, 1.0, 2.0], vec![2, 2]);
        let rhs = QnnFloatTensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
        let got = cpu_matmul(&lhs, &rhs);
        assert_eq!(got.data, vec![0.0, 0.0, 13.0, 16.0]);
    }

    /// Runs each NPU op on a shape large enough to dispatch, and checks it
    /// against the Flex reference. Ignored by default: it needs a Hexagon.
    #[cfg(qnn_sdk)]
    #[test]
    #[ignore = "needs a Snapdragon device with the HTP"]
    fn npu_ops_match_cpu_reference() {
        use burn_flex::Flex;
        use burn_tensor::{Tensor, TensorData};
        let dev = Default::default();
        let to_t = |v: &QnnFloatTensor, shape: &[usize]| -> Tensor<Flex, 4> {
            Tensor::from_data(TensorData::new(v.data.clone(), shape.to_vec()), &dev)
        };
        let before = npu_dispatch_count();

        let x = QnnFloatTensor::new(seeded(1 * 16 * 32 * 32, 7), vec![1, 16, 32, 32]);
        let w = QnnFloatTensor::new(seeded(32 * 16 * 3 * 3, 8), vec![32, 16, 3, 3]);
        let b = QnnFloatTensor::new(seeded(32, 9), vec![32]);
        let got = try_conv2d(&x, &w, Some(&b), [1, 1], [1, 1], [1, 1], 1).expect("conv on NPU");
        let want = burn_tensor::module::conv2d(
            to_t(&x, &[1, 16, 32, 32]),
            to_t(&w, &[32, 16, 3, 3]),
            Some(Tensor::<Flex, 1>::from_data(
                TensorData::new(b.data.clone(), vec![32]),
                &dev,
            )),
            burn_tensor::ops::ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
        );
        let want_v = want.into_data().convert::<f32>().into_vec::<f32>().unwrap();
        assert_close("conv2d", &got.data, &want_v);

        let p = try_pool2d(&x, true, [2, 2], [2, 2], [0, 0], false).expect("maxpool on NPU");
        assert_eq!(p.shape, vec![1, 16, 16, 16]);

        let r = try_relu(&x).expect("relu on NPU");
        let want_r: Vec<f32> = x.data.iter().map(|v| v.max(0.0)).collect();
        assert_close("relu", &r.data, &want_r);

        let c = try_concat(&[&x, &x], 1).expect("concat on NPU");
        assert_eq!(c.shape, vec![1, 32, 32, 32]);

        assert!(
            npu_dispatch_count() >= before + 4,
            "NPU counter did not move"
        );
    }

    fn assert_close(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what} length");
        let worst = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "{what} max abs diff {worst}");
    }

    #[test]
    fn matmul_falls_back_to_cpu_without_sdk() {
        // Without cfg(qnn_sdk) this is the only path; with it, qnn_matmul
        // declines small shapes and the result must still be correct.
        let lhs = QnnFloatTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let rhs = QnnFloatTensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        assert_eq!(matmul(&lhs, &rhs).data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
