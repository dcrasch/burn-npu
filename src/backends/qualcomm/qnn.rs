//! Hexagon NPU dispatch via the Qualcomm AI Engine Direct (QNN) C API.
//!
//! # Status
//!
//! **Implemented but not yet validated on hardware.** This module compiles only
//! when `build.rs` found a QAIRT SDK (`cfg(qnn_sdk)`), which requires
//! `QNN_SDK_ROOT` to be set at build time. It has been written against the
//! documented QNN C API but has never been compiled against real headers nor
//! executed on a Snapdragon device, because neither was available to the
//! author. Treat the first build on an SDK machine as the real bring-up: in
//! particular, the versioned-union accessors in [`interface_v1`] and
//! [`tensor_v1_mut`] are the places where bindgen's generated field names are
//! most likely to need adjusting.
//!
//! Every failure path returns [`QnnUnavailable`] so the caller falls back to
//! CPU. Nothing here can make the backend produce wrong answers; the worst case
//! is that it never leaves the CPU path.
//!
//! # Design
//!
//! Mirrors the OpenVINO integration: the backend library is opened at runtime
//! (so a build without a Snapdragon device still runs), one backend/device/
//! context is created lazily per process, and finalized graphs are cached by
//! matmul shape so repeated calls skip graph construction.

use super::{OpParam, QnnFloatTensor, QnnUnavailable};
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{atomic::{AtomicUsize, Ordering}, LazyLock, Mutex};

/// Bindings generated from the SDK headers by `build.rs`.
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    clippy::all
)]
mod sys {
    include!(concat!(env!("OUT_DIR"), "/qnn_bindings.rs"));
}

/// Candidate file names for the HTP (Hexagon Tensor Processor) backend library.
///
/// HTP is the NPU proper. The CPU backend is deliberately not in this list: if
/// HTP cannot be opened we want our own fallback, which keeps data in the
/// tensor representation the rest of the backend already uses.
const HTP_LIBRARIES: &[&str] = &["libQnnHtp.so", "QnnHtp.dll", "libQnnHtp.dylib"];

/// A finalized graph plus the tensor IDs QNN assigned to its inputs and output.
#[derive(Clone, Copy)]
struct CachedGraph {
    handle: sys::Qnn_GraphHandle_t,
    ids: [u32; 3],
}

/// A finalized single-op graph: its input and output tensor IDs.
#[derive(Clone)]
struct OpGraph {
    handle: sys::Qnn_GraphHandle_t,
    inputs: Vec<u32>,
    output: u32,
}

/// Stable text form of a parameter list, used as part of the graph cache key.
fn params_key(params: &[OpParam]) -> String {
    params
        .iter()
        .map(|p| match p {
            OpParam::U32(name, v) => format!("{name}={v}"),
            OpParam::F32(name, v) => format!("{name}={v:?}"),
            OpParam::Bool(name, v) => format!("{name}={v}"),
            OpParam::U32Tensor(name, _, v) => format!("{name}={v:?}"),
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// A loaded QNN backend, ready to build graphs.
struct QnnRuntime {
    /// Function-pointer table taken from the selected provider.
    iface: sys::QNN_INTERFACE_VER_TYPE,
    /// Held for the process lifetime: the context below is only valid while
    /// its backend and device are alive. Never read after construction, since
    /// `RUNTIME` is a `static` that is never dropped.
    #[allow(dead_code)]
    backend: sys::Qnn_BackendHandle_t,
    #[allow(dead_code)]
    device: sys::Qnn_DeviceHandle_t,
    context: sys::Qnn_ContextHandle_t,
    /// Finalized matmul graphs, keyed by `(m, k, n)`.
    graphs: Mutex<HashMap<(usize, usize, usize), CachedGraph>>,
    /// Finalized single-op graphs (conv, pool, ...), keyed by op and shapes.
    op_graphs: Mutex<HashMap<String, OpGraph>>,
    /// Finalized fused graphs, keyed by the caller's graph name.
    fused: Mutex<HashMap<&'static str, Option<FusedHandle>>>,
    /// Keeps the backend library mapped for as long as the handles live.
    _lib: libloading::Library,
}

// The QNN handles are opaque pointers owned by this struct and only reached
// through `&self` plus the graph mutex; the underlying API is documented as
// thread-safe for graph execution on a shared context.
unsafe impl Send for QnnRuntime {}
unsafe impl Sync for QnnRuntime {}

static RUNTIME: LazyLock<Option<QnnRuntime>> = LazyLock::new(|| match QnnRuntime::load() {
    Ok(rt) => Some(rt),
    Err(()) => None,
});

static FUSED_GRAPH_BUILD_ID: AtomicUsize = AtomicUsize::new(0);

/// Global build counter for single-op and matmul graphs. Unlike
/// [`QnnRuntime::op_graphs`]'s `len()`, this advances even when a build fails
/// partway and is retried, so a retried graph never reuses a graph name the
/// DSP has already registered (which triggers `Graph name is a duplicate`).
static OP_GRAPH_BUILD_ID: AtomicUsize = AtomicUsize::new(0);

/// Extract the v1 function table from a provider's versioned union.
///
/// The QNN headers expose this through the `QNN_INTERFACE_VER_NAME` macro,
/// which bindgen cannot follow. If the field name below does not match the
/// generated bindings, this is the single place to correct it.
///
/// # Safety
/// `provider` must point at a live `QnnInterface_t` returned by
/// `QnnInterface_getProviders`.
unsafe fn interface_v1(provider: *const sys::QnnInterface_t) -> sys::QNN_INTERFACE_VER_TYPE {
    // The union member that holds the function table is named after the QNN
    // API version (`v2_29`, `v2_31`, ...), and that name changes with every
    // SDK release. The union has exactly one member, at offset zero, so read
    // it through the union pointer instead of hard-coding a version name.
    unsafe {
        std::ptr::addr_of!((*provider).__bindgen_anon_1)
            .cast::<sys::QNN_INTERFACE_VER_TYPE>()
            .read()
    }
}

/// Mutable view of a tensor's v1 fields, for the same reason as above.
///
/// # Safety
/// `tensor` must point at a live, zero-initialised `Qnn_Tensor_t`.
unsafe fn tensor_v1_mut(tensor: *mut sys::Qnn_Tensor_t) -> *mut sys::Qnn_TensorV1_t {
    unsafe { &raw mut (*tensor).__bindgen_anon_1.v1 }
}

impl QnnRuntime {
    fn load() -> Result<Self, ()> {
        // Bare names work on desktop and in most Android processes. The app
        // also sets ADSP_LIBRARY_PATH (for the Hexagon skels), so use it as a
        // fallback for platforms whose dlopen does not search the APK's
        // native library directory.
        let mut candidates: Vec<std::path::PathBuf> =
            HTP_LIBRARIES.iter().map(std::path::PathBuf::from).collect();
        if let Ok(paths) = std::env::var("ADSP_LIBRARY_PATH") {
            for dir in paths.split(':').filter(|dir| !dir.is_empty()) {
                for name in HTP_LIBRARIES {
                    let candidate = std::path::Path::new(dir).join(name);
                    if candidate.is_file() {
                        candidates.push(candidate);
                    }
                }
            }
        }

        let lib = candidates
            .iter()
            .find_map(|path| unsafe { libloading::Library::new(path) }.ok())
            .ok_or(())?;

        // SAFETY: the symbol's signature is fixed by the QNN ABI and the
        // generated bindings describe the types it operates on.
        let get_providers: libloading::Symbol<
            unsafe extern "C" fn(*mut *const *const sys::QnnInterface_t, *mut u32) -> u64,
        > = unsafe { lib.get(b"QnnInterface_getProviders\0") }.map_err(|_| ())?;

        let mut providers: *const *const sys::QnnInterface_t = std::ptr::null();
        let mut count: u32 = 0;
        let status = unsafe { get_providers(&mut providers, &mut count) };
        if status != sys::QNN_SUCCESS as u64 || providers.is_null() || count == 0 {
            return Err(());
        }

        // Pick the first provider whose core API major version matches what we
        // generated bindings for. A major mismatch means the struct layouts
        // differ and using the table would be undefined behaviour.
        let iface = (0..count as usize)
            .filter_map(|i| {
                let provider = unsafe { *providers.add(i) };
                if provider.is_null() {
                    return None;
                }
                let version = unsafe { (*provider).apiVersion.coreApiVersion };
                (version.major == sys::QNN_API_VERSION_MAJOR)
                    .then(|| unsafe { interface_v1(provider) })
            })
            .next()
            .ok_or(())?;

        let mut backend: sys::Qnn_BackendHandle_t = std::ptr::null_mut();
        let mut device: sys::Qnn_DeviceHandle_t = std::ptr::null_mut();
        let mut context: sys::Qnn_ContextHandle_t = std::ptr::null_mut();

        unsafe {
            let create_backend = iface.backendCreate.ok_or(())?;
            if create_backend(std::ptr::null_mut(), std::ptr::null_mut(), &mut backend)
                != sys::QNN_SUCCESS as u64
            {
                return Err(());
            }

            // A device handle is optional on some backends; tolerate failure
            // and pass null through to contextCreate.
            if let Some(create_device) = iface.deviceCreate {
                if create_device(std::ptr::null_mut(), std::ptr::null_mut(), &mut device)
                    != sys::QNN_SUCCESS as u64
                {
                    device = std::ptr::null_mut();
                }
            }

            let create_context = iface.contextCreate.ok_or(())?;
            if create_context(backend, device, std::ptr::null_mut(), &mut context)
                != sys::QNN_SUCCESS as u64
            {
                return Err(());
            }
        }

        Ok(Self {
            iface,
            backend,
            device,
            context,
            graphs: Mutex::new(HashMap::new()),
            op_graphs: Mutex::new(HashMap::new()),
            fused: Mutex::new(HashMap::new()),
            _lib: lib,
        })
    }
}

/// Fill in a float32 `Qnn_Tensor_t` describing one matmul operand.
///
/// # Safety
/// `tensor` must point at a zero-initialised `Qnn_Tensor_t` that outlives the
/// graph call it is passed to, as must `name` and `dims`.
unsafe fn describe_tensor(
    tensor: *mut sys::Qnn_Tensor_t,
    name: &CString,
    dims: &mut [u32],
    tensor_type: sys::Qnn_TensorType_t,
) {
    unsafe {
        (*tensor).version = sys::Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1;
        let v1 = tensor_v1_mut(tensor);
        (*v1).id = 0;
        (*v1).name = name.as_ptr();
        (*v1).type_ = tensor_type;
        (*v1).dataFormat = sys::Qnn_TensorDataFormat_t::default();
        (*v1).dataType = sys::Qnn_DataType_t_QNN_DATATYPE_FLOAT_32;
        (*v1).quantizeParams.encodingDefinition = sys::Qnn_Definition_t_QNN_DEFINITION_DEFINED;
        (*v1).quantizeParams.quantizationEncoding =
            sys::Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED;
        (*v1).rank = dims.len() as u32;
        (*v1).dimensions = dims.as_mut_ptr();
        (*v1).memType = sys::Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW;
    }
}

/// Point a tensor's client buffer at `data`.
///
/// # Safety
/// `tensor` must be a tensor already filled in by [`describe_tensor`], and
/// `data` must outlive the graph execution.
unsafe fn set_client_buf(tensor: *mut sys::Qnn_Tensor_t, data: &mut [f32]) {
    unsafe {
        let v1 = tensor_v1_mut(tensor);
        // `clientBuf` and `memHandle` share an unnamed union in Qnn_TensorV1_t.
        let buf = &mut (*v1).__bindgen_anon_1.clientBuf;
        buf.data = data.as_mut_ptr().cast();
        buf.dataSize = std::mem::size_of_val(data) as u32;
    }
}

/// Run a 2D or batched matmul on the Hexagon NPU.
///
/// Batched inputs are executed as one graph per 2D slice, matching the shape
/// key used for caching. Returns [`QnnUnavailable`] whenever the runtime, the
/// device, or graph construction is not usable, so the caller falls back to
/// CPU.
pub fn qnn_matmul(
    lhs: &QnnFloatTensor,
    rhs: &QnnFloatTensor,
) -> Result<QnnFloatTensor, QnnUnavailable> {
    let rt = RUNTIME.as_ref().ok_or(QnnUnavailable)?;

    let lhs_ndim = lhs.shape.len();
    let rhs_ndim = rhs.shape.len();
    if lhs_ndim < 2 || rhs_ndim < 2 {
        return Err(QnnUnavailable);
    }

    let m = lhs.shape[lhs_ndim - 2];
    let k = lhs.shape[lhs_ndim - 1];
    let n = rhs.shape[rhs_ndim - 1];
    if rhs.shape[rhs_ndim - 2] != k {
        return Err(QnnUnavailable);
    }

    // Graph setup dominates for small problems; leave those on the CPU. Same
    // threshold as the OpenVINO path.
    if m * k * n < 4096 {
        return Err(QnnUnavailable);
    }

    let lhs_batch: Vec<usize> = lhs.shape[..lhs_ndim - 2].to_vec();
    let rhs_batch: Vec<usize> = rhs.shape[..rhs_ndim - 2].to_vec();
    if lhs_batch != rhs_batch {
        return Err(QnnUnavailable);
    }
    let batch: usize = lhs_batch.iter().product::<usize>().max(1);

    let graph = graph_for(rt, m, k, n)?;

    let mut out = vec![0.0f32; batch * m * n];
    for b in 0..batch {
        execute_slice(
            rt,
            graph,
            &lhs.data[b * m * k..(b + 1) * m * k],
            &rhs.data[b * k * n..(b + 1) * k * n],
            &mut out[b * m * n..(b + 1) * m * n],
            (m, k, n),
        )?;
    }

    let mut shape = lhs_batch;
    shape.push(m);
    shape.push(n);
    Ok(QnnFloatTensor::new(out, shape))
}

/// Build and finalize a matmul graph for one shape, or return the cached one.
fn graph_for(rt: &QnnRuntime, m: usize, k: usize, n: usize) -> Result<CachedGraph, QnnUnavailable> {
    let mut cache = rt.graphs.lock().map_err(|_| QnnUnavailable)?;
    if let Some(cached) = cache.get(&(m, k, n)) {
        return Ok(*cached);
    }

    let build_id = OP_GRAPH_BUILD_ID.fetch_add(1, Ordering::Relaxed);
    let graph_name =
        CString::new(format!("matmul_{m}x{k}x{n}_{build_id}")).map_err(|_| QnnUnavailable)?;
    let lhs_name = CString::new("lhs").map_err(|_| QnnUnavailable)?;
    let rhs_name = CString::new("rhs").map_err(|_| QnnUnavailable)?;
    let out_name = CString::new("out").map_err(|_| QnnUnavailable)?;
    let node_name = CString::new(format!("matmul_{build_id}")).map_err(|_| QnnUnavailable)?;
    // Every built-in QNN op lives in this package.
    let package = CString::new("qti.aisw").map_err(|_| QnnUnavailable)?;
    let op_type = CString::new("MatMul").map_err(|_| QnnUnavailable)?;

    let mut lhs_dims = [m as u32, k as u32];
    let mut rhs_dims = [k as u32, n as u32];
    let mut out_dims = [m as u32, n as u32];

    let mut graph: sys::Qnn_GraphHandle_t = std::ptr::null_mut();
    let mut ids = [0u32; 3];

    unsafe {
        let create_graph = rt.iface.graphCreate.ok_or(QnnUnavailable)?;
        if create_graph(
            rt.context,
            graph_name.as_ptr(),
            std::ptr::null_mut(),
            &mut graph,
        ) != sys::QNN_SUCCESS as u64
        {
            return Err(QnnUnavailable);
        }

        let mut lhs_t: sys::Qnn_Tensor_t = std::mem::zeroed();
        let mut rhs_t: sys::Qnn_Tensor_t = std::mem::zeroed();
        let mut out_t: sys::Qnn_Tensor_t = std::mem::zeroed();

        describe_tensor(
            &mut lhs_t,
            &lhs_name,
            &mut lhs_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        );
        describe_tensor(
            &mut rhs_t,
            &rhs_name,
            &mut rhs_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        );
        describe_tensor(
            &mut out_t,
            &out_name,
            &mut out_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
        );

        let create_tensor = rt.iface.tensorCreateGraphTensor.ok_or(QnnUnavailable)?;
        for t in [&mut lhs_t, &mut rhs_t, &mut out_t] {
            if create_tensor(graph, t) != sys::QNN_SUCCESS as u64 {
                return Err(QnnUnavailable);
            }
        }

        let (lhs_t_id, rhs_t_id, out_t_id) = (
            lhs_t.__bindgen_anon_1.v1.id,
            rhs_t.__bindgen_anon_1.v1.id,
            out_t.__bindgen_anon_1.v1.id,
        );
        ids = [
            tensor_v1_mut(&mut lhs_t).as_ref().unwrap().id,
            tensor_v1_mut(&mut rhs_t).as_ref().unwrap().id,
            tensor_v1_mut(&mut out_t).as_ref().unwrap().id,
        ];
        let mut inputs = [lhs_t, rhs_t];
        let mut outputs = [out_t];

        let mut op: sys::Qnn_OpConfig_t = std::mem::zeroed();
        op.version = sys::Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1;
        let op_v1 = &raw mut op.__bindgen_anon_1.v1;
        (*op_v1).name = node_name.as_ptr();
        (*op_v1).packageName = package.as_ptr();
        (*op_v1).typeName = op_type.as_ptr();
        (*op_v1).numOfParams = 0;
        (*op_v1).params = std::ptr::null_mut();
        (*op_v1).numOfInputs = inputs.len() as u32;
        (*op_v1).inputTensors = inputs.as_mut_ptr();
        (*op_v1).numOfOutputs = outputs.len() as u32;
        (*op_v1).outputTensors = outputs.as_mut_ptr();

        let add_node = rt.iface.graphAddNode.ok_or(QnnUnavailable)?;
        if add_node(graph, op) != sys::QNN_SUCCESS as u64 {
            return Err(QnnUnavailable);
        }

        let finalize = rt.iface.graphFinalize.ok_or(QnnUnavailable)?;
        if finalize(graph, std::ptr::null_mut(), std::ptr::null_mut()) != sys::QNN_SUCCESS as u64 {
            return Err(QnnUnavailable);
        }
    }

    let cached = CachedGraph { handle: graph, ids };
    cache.insert((m, k, n), cached);
    Ok(cached)
}

/// Execute one finalized graph over a single 2D slice.
fn execute_slice(
    rt: &QnnRuntime,
    graph: CachedGraph,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    (m, k, n): (usize, usize, usize),
) -> Result<(), QnnUnavailable> {
    let lhs_name = CString::new("lhs").map_err(|_| QnnUnavailable)?;
    let rhs_name = CString::new("rhs").map_err(|_| QnnUnavailable)?;
    let out_name = CString::new("out").map_err(|_| QnnUnavailable)?;

    let mut lhs_dims = [m as u32, k as u32];
    let mut rhs_dims = [k as u32, n as u32];
    let mut out_dims = [m as u32, n as u32];

    // QNN writes through the client buffers, so the inputs need to be owned
    // and mutable for the duration of the call.
    let mut lhs_buf = lhs.to_vec();
    let mut rhs_buf = rhs.to_vec();

    unsafe {
        let mut lhs_t: sys::Qnn_Tensor_t = std::mem::zeroed();
        let mut rhs_t: sys::Qnn_Tensor_t = std::mem::zeroed();
        let mut out_t: sys::Qnn_Tensor_t = std::mem::zeroed();

        describe_tensor(
            &mut lhs_t,
            &lhs_name,
            &mut lhs_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        );
        describe_tensor(
            &mut rhs_t,
            &rhs_name,
            &mut rhs_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        );
        describe_tensor(
            &mut out_t,
            &out_name,
            &mut out_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
        );

        set_client_buf(&mut lhs_t, &mut lhs_buf);
        set_client_buf(&mut rhs_t, &mut rhs_buf);
        set_client_buf(&mut out_t, out);
        tensor_v1_mut(&mut lhs_t).as_mut().unwrap().id = graph.ids[0];
        tensor_v1_mut(&mut rhs_t).as_mut().unwrap().id = graph.ids[1];
        tensor_v1_mut(&mut out_t).as_mut().unwrap().id = graph.ids[2];

        let inputs = [lhs_t, rhs_t];
        let mut outputs = [out_t];

        let execute = rt.iface.graphExecute.ok_or(QnnUnavailable)?;
        if execute(
            graph.handle,
            inputs.as_ptr(),
            inputs.len() as u32,
            outputs.as_mut_ptr(),
            outputs.len() as u32,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ) != sys::QNN_SUCCESS as u64
        {
            return Err(QnnUnavailable);
        }
    }

    Ok(())
}

/// Build and finalize a single-op graph, or return the cached one.
///
/// `inputs` and `output` are the static shapes of the op's data tensors. The
/// op type and parameters come from the QNN built-in `qti.aisw` package.
fn op_graph_for(
    rt: &QnnRuntime,
    key: String,
    op_type: &str,
    inputs: &[Vec<u32>],
    output: &[u32],
    params: &[OpParam],
) -> Result<OpGraph, QnnUnavailable> {
    let mut cache = rt.op_graphs.lock().map_err(|_| QnnUnavailable)?;
    if let Some(cached) = cache.get(&key) {
        return Ok(cached.clone());
    }

    let build_id = OP_GRAPH_BUILD_ID.fetch_add(1, Ordering::Relaxed);
    let graph_name = CString::new(format!("{op_type}_{build_id}")).map_err(|_| QnnUnavailable)?;
    let node_name = CString::new(format!("{op_type}_{build_id}")).map_err(|_| QnnUnavailable)?;
    let package = CString::new("qti.aisw").map_err(|_| QnnUnavailable)?;
    let type_name = CString::new(op_type).map_err(|_| QnnUnavailable)?;
    let input_names: Vec<CString> = (0..inputs.len())
        .map(|i| CString::new(format!("in{i}")).map_err(|_| QnnUnavailable))
        .collect::<Result<_, _>>()?;
    let out_name = CString::new("out").map_err(|_| QnnUnavailable)?;

    let mut input_dims: Vec<Vec<u32>> = inputs.to_vec();
    let mut output_dims: Vec<u32> = output.to_vec();

    let mut graph: sys::Qnn_GraphHandle_t = std::ptr::null_mut();
    let mut input_ids = Vec::with_capacity(inputs.len());
    let mut output_id = 0u32;

    unsafe {
        let create_graph = rt.iface.graphCreate.ok_or(QnnUnavailable)?;
        if create_graph(
            rt.context,
            graph_name.as_ptr(),
            std::ptr::null_mut(),
            &mut graph,
        ) != sys::QNN_SUCCESS as u64
        {
            return Err(QnnUnavailable);
        }

        let create_tensor = rt.iface.tensorCreateGraphTensor.ok_or(QnnUnavailable)?;
        let mut in_tensors: Vec<sys::Qnn_Tensor_t> = Vec::with_capacity(inputs.len());
        for (i, dims) in input_dims.iter_mut().enumerate() {
            let mut t: sys::Qnn_Tensor_t = std::mem::zeroed();
            describe_tensor(
                &mut t,
                &input_names[i],
                dims,
                sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
            );
            if create_tensor(graph, &mut t) != sys::QNN_SUCCESS as u64 {
                return Err(QnnUnavailable);
            }
            input_ids.push(tensor_v1_mut(&mut t).as_ref().unwrap().id);
            in_tensors.push(t);
        }

        let mut out_t: sys::Qnn_Tensor_t = std::mem::zeroed();
        describe_tensor(
            &mut out_t,
            &out_name,
            &mut output_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
        );
        if create_tensor(graph, &mut out_t) != sys::QNN_SUCCESS as u64 {
            return Err(QnnUnavailable);
        }
        output_id = tensor_v1_mut(&mut out_t).as_ref().unwrap().id;
        let mut out_tensors = [out_t];

        let mut sys_params: Vec<sys::Qnn_Param_t> = Vec::with_capacity(params.len());
        let mut param_names: Vec<CString> = Vec::with_capacity(params.len());
        let mut param_dims: Vec<Vec<u32>> = Vec::new();
        let mut param_vals: Vec<Vec<u32>> = Vec::new();
        for p in params {
            let (name, kind) = match p {
                OpParam::U32(n, _) | OpParam::F32(n, _) | OpParam::Bool(n, _) => (*n, 0),
                OpParam::U32Tensor(n, _, _) => (*n, 1),
            };
            param_names.push(CString::new(name).map_err(|_| QnnUnavailable)?);
            let mut sp: sys::Qnn_Param_t = std::mem::zeroed();
            sp.name = param_names.last().unwrap().as_ptr();
            if kind == 0 {
                sp.paramType = sys::Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR;
                let scalar = &mut sp.__bindgen_anon_1.scalarParam;
                match p {
                    OpParam::U32(_, v) => {
                        scalar.dataType = sys::Qnn_DataType_t_QNN_DATATYPE_UINT_32;
                        scalar.__bindgen_anon_1.uint32Value = *v;
                    }
                    OpParam::F32(_, v) => {
                        scalar.dataType = sys::Qnn_DataType_t_QNN_DATATYPE_FLOAT_32;
                        scalar.__bindgen_anon_1.floatValue = *v;
                    }
                    OpParam::Bool(_, v) => {
                        scalar.dataType = sys::Qnn_DataType_t_QNN_DATATYPE_BOOL_8;
                        scalar.__bindgen_anon_1.bool8Value = *v as u8;
                    }
                    OpParam::U32Tensor(..) => unreachable!(),
                }
            } else {
                // Tensor params are static, UINT_32, and must be built as
                // Qnn_Tensor_t with client data; the dims live in `param_dims`.
                let OpParam::U32Tensor(_, dims, vals) = p else {
                    unreachable!()
                };
                param_dims.push(dims.clone());
                param_vals.push(vals.clone());
                sp.paramType = sys::Qnn_ParamType_t_QNN_PARAMTYPE_TENSOR;
                let t = &mut sp.__bindgen_anon_1.tensorParam;
                t.version = sys::Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1;
                let v1 = tensor_v1_mut(t);
                (*v1).name = param_names.last().unwrap().as_ptr();
                (*v1).type_ = sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_STATIC;
                (*v1).dataFormat = sys::QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
                (*v1).dataType = sys::Qnn_DataType_t_QNN_DATATYPE_UINT_32;
                // Match the SDK's model builder: uint32 params are not
                // quantized, so mark their encoding undefined. Leaving this
                // zeroed makes the DSP see QNN_DEFINITION_IMPL_GENERATED and
                // reject the op with "Failed to create InputDef ... param".
                (*v1).quantizeParams.encodingDefinition =
                    sys::Qnn_Definition_t_QNN_DEFINITION_UNDEFINED;
                (*v1).quantizeParams.quantizationEncoding =
                    sys::Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED;
                (*v1).rank = dims.len() as u32;
                (*v1).dimensions = param_dims.last_mut().unwrap().as_mut_ptr();
                (*v1).memType = sys::Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW;
                let buf = &mut (*v1).__bindgen_anon_1.clientBuf;
                buf.data = param_vals.last_mut().unwrap().as_mut_ptr().cast();
                buf.dataSize = (vals.len() * std::mem::size_of::<u32>()) as u32;
                // The HTP/DSP backend requires every tensor param to first be
                // registered as a graph tensor (the SDK's QnnModel::addNode
                // does this via addTensor, which calls createGraphTensor).
                // Without it, graphAddNode fails with "Failed to create
                // InputDef for Op ... param <name>". createGraphTensor also
                // overwrites `id` with the value the node must reference.
                if create_tensor(graph, t) != sys::QNN_SUCCESS as u64 {
                    return Err(QnnUnavailable);
                }
            }
            sys_params.push(sp);
        }

        let mut op: sys::Qnn_OpConfig_t = std::mem::zeroed();
        op.version = sys::Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1;
        let op_v1 = &raw mut op.__bindgen_anon_1.v1;
        (*op_v1).name = node_name.as_ptr();
        (*op_v1).packageName = package.as_ptr();
        (*op_v1).typeName = type_name.as_ptr();
        (*op_v1).numOfParams = sys_params.len() as u32;
        (*op_v1).params = sys_params.as_mut_ptr();
        (*op_v1).numOfInputs = in_tensors.len() as u32;
        (*op_v1).inputTensors = in_tensors.as_mut_ptr();
        (*op_v1).numOfOutputs = out_tensors.len() as u32;
        (*op_v1).outputTensors = out_tensors.as_mut_ptr();

        let add_node = rt.iface.graphAddNode.ok_or(QnnUnavailable)?;
        if add_node(graph, op) != sys::QNN_SUCCESS as u64 {
            return Err(QnnUnavailable);
        }

        let finalize = rt.iface.graphFinalize.ok_or(QnnUnavailable)?;
        if finalize(graph, std::ptr::null_mut(), std::ptr::null_mut()) != sys::QNN_SUCCESS as u64 {
            return Err(QnnUnavailable);
        }
    }

    let cached = OpGraph {
        handle: graph,
        inputs: input_ids,
        output: output_id,
    };
    cache.insert(key, cached.clone());
    Ok(cached)
}

/// Execute a finalized single-op graph on NHWC-or-flat f32 inputs.
fn op_execute(
    rt: &QnnRuntime,
    graph: &OpGraph,
    inputs: &[(&[u32], &[f32])],
    output_dims: &[u32],
    output: &mut [f32],
) -> Result<(), QnnUnavailable> {
    let input_names: Vec<CString> = (0..inputs.len())
        .map(|i| CString::new(format!("in{i}")).map_err(|_| QnnUnavailable))
        .collect::<Result<_, _>>()?;
    let out_name = CString::new("out").map_err(|_| QnnUnavailable)?;

    let mut in_dims: Vec<Vec<u32>> = inputs.iter().map(|(d, _)| d.to_vec()).collect();
    let mut out_dims = output_dims.to_vec();

    unsafe {
        let mut in_tensors: Vec<sys::Qnn_Tensor_t> = Vec::with_capacity(inputs.len());
        for i in 0..inputs.len() {
            let mut t: sys::Qnn_Tensor_t = std::mem::zeroed();
            describe_tensor(
                &mut t,
                &input_names[i],
                &mut in_dims[i],
                sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
            );
            // QNN only reads input client buffers, so the caller's slice is
            // passed directly instead of being copied.
            set_client_buf_const(&mut t, inputs[i].1);
            tensor_v1_mut(&mut t).as_mut().unwrap().id = graph.inputs[i];
            in_tensors.push(t);
        }

        let mut out_t: sys::Qnn_Tensor_t = std::mem::zeroed();
        describe_tensor(
            &mut out_t,
            &out_name,
            &mut out_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
        );
        set_client_buf(&mut out_t, output);
        tensor_v1_mut(&mut out_t).as_mut().unwrap().id = graph.output;
        let mut out_tensors = [out_t];

        let execute = rt.iface.graphExecute.ok_or(QnnUnavailable)?;
        if execute(
            graph.handle,
            in_tensors.as_ptr(),
            in_tensors.len() as u32,
            out_tensors.as_mut_ptr(),
            out_tensors.len() as u32,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ) != sys::QNN_SUCCESS as u64
        {
            return Err(QnnUnavailable);
        }
    }

    Ok(())
}

/// Per-op timing is printed when `BURN_NPU_TIMING` is set in the environment
/// or the `debug.burnnpu.timing` system property is `1` on Android.
fn timing_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        if std::env::var_os("BURN_NPU_TIMING").is_some() {
            return true;
        }
        std::process::Command::new("getprop")
            .arg("debug.burnnpu.timing")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "1")
            .unwrap_or(false)
    });
    *ENABLED
}

/// Run one op: build or fetch its graph, then execute it.
fn run_op(
    rt: &QnnRuntime,
    op_type: &str,
    inputs: &[(Vec<u32>, &[f32])],
    out_dims: Vec<u32>,
    params: &[OpParam],
) -> Result<Vec<f32>, QnnUnavailable> {
    let in_dims: Vec<Vec<u32>> = inputs.iter().map(|(d, _)| d.clone()).collect();
    let key = format!("{op_type}|{in_dims:?}|{out_dims:?}|{}", params_key(params));
    let start = std::time::Instant::now();
    let graph = op_graph_for(rt, key, op_type, &in_dims, &out_dims, params)?;
    let built = start.elapsed();
    super::record_build_time(built.as_nanos() as u64);
    let mut out = vec![0.0f32; out_dims.iter().map(|&d| d as usize).product()];
    let refs: Vec<(&[u32], &[f32])> = inputs.iter().map(|(d, v)| (d.as_slice(), *v)).collect();
    let exec_start = std::time::Instant::now();
    op_execute(rt, &graph, &refs, &out_dims, &mut out)?;
    let exec = exec_start.elapsed();
    super::record_exec_time(exec.as_nanos() as u64);
    if timing_enabled() {
        eprintln!(
            "npu-op {op_type} {in_dims:?} -> {out_dims:?} build {:.3} ms exec {:.3} ms",
            built.as_secs_f64() * 1000.0,
            exec.as_secs_f64() * 1000.0
        );
    }
    Ok(out)
}

/// Convert an NCHW buffer to NHWC.
fn nchw_to_nhwc(data: &[f32], n: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for b in 0..n {
        for ci in 0..c {
            for y in 0..h {
                for x in 0..w {
                    out[((b * h + y) * w + x) * c + ci] = data[((b * c + ci) * h + y) * w + x];
                }
            }
        }
    }
    out
}

/// Convert an NHWC buffer to NCHW.
fn nhwc_to_nchw(data: &[f32], n: usize, h: usize, w: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for b in 0..n {
        for y in 0..h {
            for x in 0..w {
                for ci in 0..c {
                    out[((b * c + ci) * h + y) * w + x] = data[((b * h + y) * w + x) * c + ci];
                }
            }
        }
    }
    out
}

/// HWIO copy of a weight, converted once per weight allocation and reused.
/// Model weights live in fixed buffers for the life of the process, so the
/// data pointer and length identify a weight without hashing its contents.
fn hwio_weight(data: &[f32], o: usize, i: usize, kh: usize, kw: usize) -> std::sync::Arc<Vec<f32>> {
    static CACHE: LazyLock<Mutex<HashMap<(usize, usize), std::sync::Arc<Vec<f32>>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let key = (data.as_ptr() as usize, data.len());
    let mut cache = CACHE.lock().expect("hwio cache poisoned");
    cache
        .entry(key)
        .or_insert_with(|| std::sync::Arc::new(oihw_to_hwio(data, o, i, kh, kw)))
        .clone()
}

/// Convert OIHW convolution weights to HWIO, the layout QNN `Conv2d` expects.
fn oihw_to_hwio(data: &[f32], o: usize, i: usize, kh: usize, kw: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for oc in 0..o {
        for ic in 0..i {
            for y in 0..kh {
                for x in 0..kw {
                    out[((y * kw + x) * i + ic) * o + oc] = data[((oc * i + ic) * kh + y) * kw + x];
                }
            }
        }
    }
    out
}

fn dims_u32(dims: &[usize]) -> Vec<u32> {
    dims.iter().map(|&d| d as u32).collect()
}

/// 2D convolution (NCHW input, OIHW weights) on the Hexagon NPU.
///
/// Returns [`QnnUnavailable`] for any shape or option this path does not
/// cover, so the caller falls back to CPU.
pub fn qnn_conv2d(
    x: &QnnFloatTensor,
    weight: &QnnFloatTensor,
    bias: Option<&QnnFloatTensor>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Result<QnnFloatTensor, QnnUnavailable> {
    let rt = RUNTIME.as_ref().ok_or(QnnUnavailable)?;
    if x.shape.len() != 4 || weight.shape.len() != 4 || groups == 0 {
        return Err(QnnUnavailable);
    }
    let [n, c, h, w] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    let [o, ci_g, kh, kw] = [
        weight.shape[0],
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
    ];
    if ci_g * groups != c || o % groups != 0 || stride.contains(&0) || dilation.contains(&0) {
        return Err(QnnUnavailable);
    }
    let eff_kh = dilation[0] * (kh - 1) + 1;
    let eff_kw = dilation[1] * (kw - 1) + 1;
    if h + 2 * padding[0] < eff_kh || w + 2 * padding[1] < eff_kw {
        return Err(QnnUnavailable);
    }
    let oh = (h + 2 * padding[0] - eff_kh) / stride[0] + 1;
    let ow = (w + 2 * padding[1] - eff_kw) / stride[1] + 1;
    if n * o * oh * ow < 4096 {
        return Err(QnnUnavailable);
    }

    let x_nhwc = nchw_to_nhwc(&x.data, n, c, h, w);
    let w_hwio = hwio_weight(&weight.data, o, ci_g, kh, kw);
    let zeros = vec![0.0f32; o];
    let bias_data: &[f32] = match bias {
        Some(b) => &b.data,
        None => &zeros,
    };

    let out = run_op(
        rt,
        "Conv2d",
        &[
            (dims_u32(&[n, h, w, c]), &x_nhwc),
            (dims_u32(&[kh, kw, ci_g, o]), &w_hwio),
            (dims_u32(&[o]), bias_data),
        ],
        dims_u32(&[n, oh, ow, o]),
        &[
            OpParam::U32Tensor("stride", vec![2], vec![stride[0] as u32, stride[1] as u32]),
            OpParam::U32Tensor(
                "pad_amount",
                vec![2, 2],
                vec![
                    padding[0] as u32,
                    padding[0] as u32,
                    padding[1] as u32,
                    padding[1] as u32,
                ],
            ),
            OpParam::U32Tensor(
                "dilation",
                vec![2],
                vec![dilation[0] as u32, dilation[1] as u32],
            ),
            OpParam::U32("group", groups as u32),
        ],
    )?;

    Ok(QnnFloatTensor::new(
        nhwc_to_nchw(&out, n, oh, ow, o),
        vec![n, o, oh, ow],
    ))
}

/// 2D max or average pooling (NCHW input) on the Hexagon NPU.
pub fn qnn_pool2d(
    x: &QnnFloatTensor,
    max: bool,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> Result<QnnFloatTensor, QnnUnavailable> {
    let rt = RUNTIME.as_ref().ok_or(QnnUnavailable)?;
    if x.shape.len() != 4 || stride.contains(&0) || kernel.contains(&0) {
        return Err(QnnUnavailable);
    }
    let [n, c, h, w] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    if h + 2 * padding[0] < kernel[0] || w + 2 * padding[1] < kernel[1] {
        return Err(QnnUnavailable);
    }
    let oh = (h + 2 * padding[0] - kernel[0]) / stride[0] + 1;
    let ow = (w + 2 * padding[1] - kernel[1]) / stride[1] + 1;
    if n * c * oh * ow < 4096 {
        return Err(QnnUnavailable);
    }

    let x_nhwc = nchw_to_nhwc(&x.data, n, c, h, w);
    let mut params = vec![
        OpParam::U32Tensor(
            "filter_size",
            vec![2],
            vec![kernel[0] as u32, kernel[1] as u32],
        ),
        OpParam::U32Tensor("stride", vec![2], vec![stride[0] as u32, stride[1] as u32]),
        OpParam::U32Tensor(
            "pad_amount",
            vec![2, 2],
            vec![
                padding[0] as u32,
                padding[0] as u32,
                padding[1] as u32,
                padding[1] as u32,
            ],
        ),
        // Floor rounding matches Burn's default (ceil_mode = false).
        OpParam::U32("rounding_mode", 0),
    ];
    let op_type = if max {
        "PoolMax2d"
    } else {
        params.push(OpParam::Bool("count_pad_for_edges", count_include_pad));
        "PoolAvg2d"
    };

    let out = run_op(
        rt,
        op_type,
        &[(dims_u32(&[n, h, w, c]), &x_nhwc)],
        dims_u32(&[n, oh, ow, c]),
        &params,
    )?;
    Ok(QnnFloatTensor::new(
        nhwc_to_nchw(&out, n, oh, ow, c),
        vec![n, c, oh, ow],
    ))
}

/// Elementwise ReLU on the Hexagon NPU.
pub fn qnn_relu(x: &QnnFloatTensor) -> Result<QnnFloatTensor, QnnUnavailable> {
    let rt = RUNTIME.as_ref().ok_or(QnnUnavailable)?;
    if x.data.len() < 4096 {
        return Err(QnnUnavailable);
    }
    let dims = dims_u32(&x.shape);
    let out = run_op(rt, "Relu", &[(dims.clone(), &x.data)], dims, &[])?;
    Ok(QnnFloatTensor::new(out, x.shape.clone()))
}

/// Concatenate NCHW tensors along `axis` on the Hexagon NPU.
pub fn qnn_concat(
    tensors: &[&QnnFloatTensor],
    axis: usize,
) -> Result<QnnFloatTensor, QnnUnavailable> {
    let rt = RUNTIME.as_ref().ok_or(QnnUnavailable)?;
    let first = tensors.first().ok_or(QnnUnavailable)?;
    let rank = first.shape.len();
    if axis >= rank
        || tensors.iter().any(|t| {
            t.shape.len() != rank
                || t.shape[..axis] != first.shape[..axis]
                || t.shape[axis + 1..] != first.shape[axis + 1..]
        })
    {
        return Err(QnnUnavailable);
    }
    let total_axis: usize = tensors.iter().map(|t| t.shape[axis]).sum();
    let mut out_shape = first.shape.clone();
    out_shape[axis] = total_axis;
    let out_numel: usize = out_shape.iter().product();
    if out_numel < 4096 {
        return Err(QnnUnavailable);
    }

    let inputs: Vec<(Vec<u32>, &[f32])> = tensors
        .iter()
        .map(|t| (dims_u32(&t.shape), t.data.as_slice()))
        .collect();
    let out = run_op(
        rt,
        "Concat",
        &inputs,
        dims_u32(&out_shape),
        &[OpParam::U32("axis", axis as u32)],
    )?;
    Ok(QnnFloatTensor::new(out, out_shape))
}

/// Softmax over the last axis on the Hexagon NPU.
pub fn qnn_softmax(x: &QnnFloatTensor) -> Result<QnnFloatTensor, QnnUnavailable> {
    let rt = RUNTIME.as_ref().ok_or(QnnUnavailable)?;
    let rank = x.shape.len();
    if rank == 0 || x.data.len() < 1024 {
        return Err(QnnUnavailable);
    }
    let dims = dims_u32(&x.shape);
    let out = run_op(
        rt,
        "Softmax",
        &[(dims.clone(), &x.data)],
        dims,
        &[
            OpParam::U32("axis", (rank - 1) as u32),
            OpParam::F32("beta", 1.0),
        ],
    )?;
    Ok(QnnFloatTensor::new(out, x.shape.clone()))
}

/// Cached finalized fused graph: its handle and the IDs of its I/O tensors.
#[derive(Clone, Copy)]
struct FusedHandle {
    handle: sys::Qnn_GraphHandle_t,
    input: u32,
    output: u32,
    output_len: usize,
}

/// Build (once per `key`) and run a fused graph. Intermediates stay on the
/// device. The description must be `'static` because QNN keeps pointers to
/// the static weights for the lifetime of the graph.
pub fn qnn_fused(
    key: &'static str,
    graph: &'static super::FusedGraph,
    input_data: &[f32],
) -> Result<Vec<f32>, QnnUnavailable> {
    let rt = RUNTIME.as_ref().ok_or(QnnUnavailable)?;
    let handle = {
        let mut cache = rt.fused.lock().map_err(|_| QnnUnavailable)?;
        match cache.get(key).copied() {
            Some(Some(h)) => h,
            Some(None) => return Err(QnnUnavailable),
            None => {
                let build_start = std::time::Instant::now();
                let built = build_fused(rt, key, graph);
                super::record_build_time(build_start.elapsed().as_nanos() as u64);
                match built {
                    Ok(h) => {
                        cache.insert(key, Some(h));
                        h
                    }
                    Err(err) => {
                        cache.insert(key, None);
                        return Err(err);
                    }
                }
            }
        }
    };

    let mut in_dims = graph.tensors[graph.input].dims.clone();
    let mut out_dims = graph.tensors[graph.output].dims.clone();
    let in_name = CString::new("in").map_err(|_| QnnUnavailable)?;
    let out_name = CString::new("out").map_err(|_| QnnUnavailable)?;
    let mut in_buf = input_data.to_vec();
    let mut out_buf = vec![0.0f32; handle.output_len];
    unsafe {
        let mut in_t: sys::Qnn_Tensor_t = std::mem::zeroed();
        describe_tensor(
            &mut in_t,
            &in_name,
            &mut in_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        );
        tensor_v1_mut(&mut in_t).as_mut().unwrap().id = handle.input;
        set_client_buf(&mut in_t, &mut in_buf);

        let mut out_t: sys::Qnn_Tensor_t = std::mem::zeroed();
        describe_tensor(
            &mut out_t,
            &out_name,
            &mut out_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
        );
        tensor_v1_mut(&mut out_t).as_mut().unwrap().id = handle.output;
        set_client_buf(&mut out_t, &mut out_buf);

        let inputs = [in_t];
        let mut outputs = [out_t];
        let execute = rt.iface.graphExecute.ok_or(QnnUnavailable)?;
        let exec_start = std::time::Instant::now();
        let status = execute(
            handle.handle,
            inputs.as_ptr(),
            1,
            outputs.as_mut_ptr(),
            1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        super::record_exec_time(exec_start.elapsed().as_nanos() as u64);
        if status != sys::QNN_SUCCESS as u64 {
            return Err(QnnUnavailable);
        }
    }
    Ok(out_buf)
}

/// Create, populate, and finalize the fused graph for `graph`.
fn build_fused(
    rt: &QnnRuntime,
    key: &str,
    graph: &'static super::FusedGraph,
) -> Result<FusedHandle, QnnUnavailable> {
    let build_id = FUSED_GRAPH_BUILD_ID.fetch_add(1, Ordering::Relaxed);
    let mut handle: sys::Qnn_GraphHandle_t = std::ptr::null_mut();
    let graph_name = CString::new(format!("{}_{}", key, build_id)).map_err(|_| QnnUnavailable)?;
    let names: Vec<CString> = (0..graph.tensors.len())
        .map(|i| CString::new(format!("t{i}")).map_err(|_| QnnUnavailable))
        .collect::<Result<_, _>>()?;
    let mut dims: Vec<Vec<u32>> = graph.tensors.iter().map(|t| t.dims.clone()).collect();
    let mut ids = vec![0u32; graph.tensors.len()];
    let input = graph.input;
    let output = graph.output;

    unsafe {
        let create_graph = rt.iface.graphCreate.ok_or(QnnUnavailable)?;
        if create_graph(
            rt.context,
            graph_name.as_ptr(),
            std::ptr::null_mut(),
            &mut handle,
        ) != sys::QNN_SUCCESS as u64
        {
            return Err(QnnUnavailable);
        }

        let create_tensor = rt.iface.tensorCreateGraphTensor.ok_or(QnnUnavailable)?;
        for (i, t) in graph.tensors.iter().enumerate() {
            let ttype = if t.data.is_some() {
                sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_STATIC
            } else if i == input {
                sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE
            } else if i == output {
                sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ
            } else {
                sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE
            };
            let mut sys_t: sys::Qnn_Tensor_t = std::mem::zeroed();
            describe_tensor(&mut sys_t, &names[i], &mut dims[i], ttype);
            if let Some(data) = &t.data {
                set_client_buf_const(&mut sys_t, data);
            }
            if create_tensor(handle, &mut sys_t) != sys::QNN_SUCCESS as u64 {
                return Err(QnnUnavailable);
            }
            ids[i] = tensor_v1_mut(&mut sys_t).as_ref().unwrap().id;
        }

        let add_node = rt.iface.graphAddNode.ok_or(QnnUnavailable)?;
        let package = CString::new("qti.aisw").map_err(|_| QnnUnavailable)?;
        // QNN reads each op's definition, parameters and tensors until the
        // graph is finalized, so keep them alive until this function returns.
        let mut kept: Vec<OpStorage> = Vec::with_capacity(graph.ops.len());
        for (index, op) in graph.ops.iter().enumerate() {
            let node_name = CString::new(format!("{key}_{}_{index}_{build_id}", op.op_type))
                .map_err(|_| QnnUnavailable)?;
            let type_name = CString::new(op.op_type).map_err(|_| QnnUnavailable)?;
            let ins: Vec<sys::Qnn_Tensor_t> = op
                .inputs
                .iter()
                .map(|&t| {
                    let mut sys_t: sys::Qnn_Tensor_t = std::mem::zeroed();
                    describe_tensor(
                        &mut sys_t,
                        &names[t],
                        &mut dims[t],
                        sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE,
                    );
                    tensor_v1_mut(&mut sys_t).as_mut().unwrap().id = ids[t];
                    sys_t
                })
                .collect();
            let mut out_t: sys::Qnn_Tensor_t = std::mem::zeroed();
            describe_tensor(
                &mut out_t,
                &names[op.output],
                &mut dims[op.output],
                sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE,
            );
            tensor_v1_mut(&mut out_t).as_mut().unwrap().id = ids[op.output];

            let (params, param_names, tensor_names, param_dims, param_vals) = build_params(
                &op.params,
                node_name.to_str().map_err(|_| QnnUnavailable)?,
            )?;
            kept.push(OpStorage {
                node_name,
                type_name,
                params,
                _param_names: param_names,
                _tensor_names: tensor_names,
                _param_dims: param_dims,
                _param_vals: param_vals,
                ins,
                outs: vec![out_t],
            });
            let storage = kept.last_mut().unwrap();
            // Register every tensor param as a graph tensor before adding the
            // node. The HTP/DSP backend rejects tensor params that were not
            // created this way ("Failed to create InputDef ... param <name>").
            for param in storage.params.iter_mut() {
                if param.paramType == sys::Qnn_ParamType_t_QNN_PARAMTYPE_TENSOR {
                    let t = &mut param.__bindgen_anon_1.tensorParam;
                    if create_tensor(handle, t) != sys::QNN_SUCCESS as u64 {
                        return Err(QnnUnavailable);
                    }
                }
            }
            let mut op_cfg: sys::Qnn_OpConfig_t = std::mem::zeroed();
            op_cfg.version = sys::Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1;
            let v1 = &raw mut op_cfg.__bindgen_anon_1.v1;
            (*v1).name = storage.node_name.as_ptr();
            (*v1).packageName = package.as_ptr();
            (*v1).typeName = storage.type_name.as_ptr();
            (*v1).numOfParams = storage.params.len() as u32;
            (*v1).params = storage.params.as_mut_ptr();
            (*v1).numOfInputs = storage.ins.len() as u32;
            (*v1).inputTensors = storage.ins.as_mut_ptr();
            (*v1).numOfOutputs = storage.outs.len() as u32;
            (*v1).outputTensors = storage.outs.as_mut_ptr();
            if add_node(handle, op_cfg) != sys::QNN_SUCCESS as u64 {
                return Err(QnnUnavailable);
            }
        }

        let finalize = rt.iface.graphFinalize.ok_or(QnnUnavailable)?;
        if finalize(handle, std::ptr::null_mut(), std::ptr::null_mut()) != sys::QNN_SUCCESS as u64 {
            return Err(QnnUnavailable);
        }
    }

    Ok(FusedHandle {
        handle,
        input: ids[input],
        output: ids[output],
        output_len: graph.tensors[output]
            .dims
            .iter()
            .map(|&d| d as usize)
            .product(),
    })
}

/// Build the QNN parameter list for one op. The returned vectors own the
/// names, dims and values that the parameters point into, so they must stay
/// alive until the op is added to the graph.
///
/// Tensor params become graph tensors, which must have unique names within the
/// graph, so `node_name` is used to disambiguate the tensor name from the param
/// name (which the op definition matches, and which may repeat across nodes).
#[allow(clippy::type_complexity)]
fn build_params(
    params: &[OpParam],
    node_name: &str,
) -> Result<
    (
        Vec<sys::Qnn_Param_t>,
        Vec<CString>,
        Vec<CString>,
        Vec<Vec<u32>>,
        Vec<Vec<u32>>,
    ),
    QnnUnavailable,
> {
    let mut sys_params: Vec<sys::Qnn_Param_t> = Vec::with_capacity(params.len());
    let mut names: Vec<CString> = Vec::with_capacity(params.len());
    let mut tensor_names: Vec<CString> = Vec::with_capacity(params.len());
    let mut dims_store: Vec<Vec<u32>> = Vec::new();
    let mut vals_store: Vec<Vec<u32>> = Vec::new();
    for p in params {
        let name = match p {
            OpParam::U32(n, _)
            | OpParam::F32(n, _)
            | OpParam::Bool(n, _)
            | OpParam::U32Tensor(n, _, _) => *n,
        };
        names.push(CString::new(name).map_err(|_| QnnUnavailable)?);
        let mut sp: sys::Qnn_Param_t = unsafe { std::mem::zeroed() };
        sp.name = names.last().unwrap().as_ptr();
        unsafe {
            match p {
                OpParam::U32(_, v) => {
                    sp.paramType = sys::Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR;
                    sp.__bindgen_anon_1.scalarParam.dataType =
                        sys::Qnn_DataType_t_QNN_DATATYPE_UINT_32;
                    sp.__bindgen_anon_1.scalarParam.__bindgen_anon_1.uint32Value = *v;
                }
                OpParam::F32(_, v) => {
                    sp.paramType = sys::Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR;
                    sp.__bindgen_anon_1.scalarParam.dataType =
                        sys::Qnn_DataType_t_QNN_DATATYPE_FLOAT_32;
                    sp.__bindgen_anon_1.scalarParam.__bindgen_anon_1.floatValue = *v;
                }
                OpParam::Bool(_, v) => {
                    sp.paramType = sys::Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR;
                    sp.__bindgen_anon_1.scalarParam.dataType =
                        sys::Qnn_DataType_t_QNN_DATATYPE_BOOL_8;
                    sp.__bindgen_anon_1.scalarParam.__bindgen_anon_1.bool8Value = *v as u8;
                }
                OpParam::U32Tensor(_, dims, vals) => {
                    dims_store.push(dims.clone());
                    vals_store.push(vals.clone());
                    // Tensor param names must be unique (they are registered
                    // as graph tensors); the bare param name is shared.
                    tensor_names.push(
                        CString::new(format!(
                            "{node_name}_{}",
                            names.last().unwrap().to_str().map_err(|_| QnnUnavailable)?
                        ))
                        .map_err(|_| QnnUnavailable)?,
                    );
                    sp.paramType = sys::Qnn_ParamType_t_QNN_PARAMTYPE_TENSOR;
                    let t = &mut sp.__bindgen_anon_1.tensorParam;
                    t.version = sys::Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1;
                    let v1 = tensor_v1_mut(t);
                    (*v1).name = tensor_names.last().unwrap().as_ptr();
                    (*v1).type_ = sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_STATIC;
                    (*v1).dataFormat = sys::QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
                    (*v1).dataType = sys::Qnn_DataType_t_QNN_DATATYPE_UINT_32;
                    // Match the SDK's model builder: uint32 params are not
                    // quantized, so mark their encoding undefined. Leaving this
                    // defined/undefined leaves the DSP confused about encoding.
                    (*v1).quantizeParams.encodingDefinition =
                        sys::Qnn_Definition_t_QNN_DEFINITION_UNDEFINED;
                    (*v1).quantizeParams.quantizationEncoding =
                        sys::Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED;
                    (*v1).rank = dims.len() as u32;
                    (*v1).dimensions = dims_store.last_mut().unwrap().as_mut_ptr();
                    (*v1).memType = sys::Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW;
                    let buf = &mut (*v1).__bindgen_anon_1.clientBuf;
                    buf.data = vals_store.last_mut().unwrap().as_mut_ptr().cast();
                    buf.dataSize = (vals.len() * std::mem::size_of::<u32>()) as u32;
                }
            }
        }
        sys_params.push(sp);
    }
    Ok((sys_params, names, tensor_names, dims_store, vals_store))
}

/// Point a static tensor at read-only data. QNN only reads static tensors.
///
/// # Safety
/// `tensor` must be filled in by [`describe_tensor`], and `data` must outlive
/// the graph.
unsafe fn set_client_buf_const(tensor: *mut sys::Qnn_Tensor_t, data: &[f32]) {
    unsafe {
        let v1 = tensor_v1_mut(tensor);
        let buf = &mut (*v1).__bindgen_anon_1.clientBuf;
        buf.data = data.as_ptr().cast_mut().cast();
        buf.dataSize = std::mem::size_of_val(data) as u32;
    }
}

/// Everything one op's QNN definition points into.
#[allow(dead_code)]
struct OpStorage {
    node_name: CString,
    type_name: CString,
    params: Vec<sys::Qnn_Param_t>,
    _param_names: Vec<CString>,
    _tensor_names: Vec<CString>,
    _param_dims: Vec<Vec<u32>>,
    _param_vals: Vec<Vec<u32>>,
    ins: Vec<sys::Qnn_Tensor_t>,
    outs: Vec<sys::Qnn_Tensor_t>,
}
