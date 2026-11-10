use crate::info::{NpuInfo, NpuVendor, Precision};

/// Probe the current hardware and return [`NpuInfo`] if an NPU is present.
///
/// Detection is platform-specific:
/// - **macOS**: checks for Apple Silicon via `sysctl hw.optional.arm64`
/// - **Linux**: checks the device tree / `/proc/cpuinfo` for Qualcomm, then
///   `/proc/cpuinfo` for Intel + `/dev/accel*` for an NPU device
/// - **Windows**: checks for the QNN HTP backend DLL, then for OpenVINO
///
/// Qualcomm is probed before Intel because the checks are mutually exclusive
/// and the Qualcomm one is cheaper.
///
/// Returns `None` if no NPU is detected or on unsupported platforms.
pub fn detect() -> Option<NpuInfo> {
    #[cfg(target_os = "macos")]
    {
        if let Some(info) = detect_apple() {
            return Some(info);
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        if let Some(info) = detect_qualcomm_linux() {
            return Some(info);
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Some(info) = detect_intel_linux() {
            return Some(info);
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(info) = detect_qualcomm_windows() {
            return Some(info);
        }
        if let Some(info) = detect_intel_windows() {
            return Some(info);
        }
    }

    None
}

// ── macOS: Apple Silicon Neural Engine ──────────────────────────────────

#[cfg(target_os = "macos")]
fn detect_apple() -> Option<NpuInfo> {
    use std::process::Command;

    let output = Command::new("sysctl")
        .args(["-n", "hw.optional.arm64"])
        .output()
        .ok()?;
    let arm64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u32>()
        .unwrap_or(0);
    if arm64 != 1 {
        return None;
    }

    let brand_output = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()?;
    let brand = String::from_utf8_lossy(&brand_output.stdout)
        .trim()
        .to_string();

    let (tops, desc) = estimate_apple_tops(&brand);

    Some(NpuInfo {
        vendor: NpuVendor::Apple,
        tops,
        max_precision: Precision::FP16,
        description: desc,
    })
}

/// Peak ANE throughput for a given Apple Silicon brand string.
///
/// Apple published TOPS figures through M4 (38 TOPS) but has not disclosed one
/// for the M5 or M6 Neural Engine, so those report the M4 figure as a
/// documented floor rather than an invented number -- they are certainly not
/// slower. An unrecognised chip gets the same floor: in practice "unknown"
/// means newer than this table, and reporting M1-era numbers for an M6 (as
/// this function previously did) is the worse failure.
///
/// Note that from M5 onward the Neural Engine is no longer the whole story:
/// Apple put a Neural Accelerator in every GPU core, and much of the headline
/// AI throughput on those parts lives there rather than on the ANE. This
/// backend targets the ANE via MLTensor, so `tops` describes the ANE alone.
#[cfg(target_os = "macos")]
fn estimate_apple_tops(brand: &str) -> (f32, String) {
    /// Apple's last published Neural Engine figure, used as a floor for
    /// anything newer.
    const PUBLISHED_FLOOR: f32 = 38.0;

    let lower = brand.to_lowercase();
    if lower.contains("m6") {
        (
            PUBLISHED_FLOOR,
            format!("Apple Neural Engine (M6, TOPS not published) -- {brand}"),
        )
    } else if lower.contains("m5") {
        (
            PUBLISHED_FLOOR,
            format!("Apple Neural Engine (M5, TOPS not published) -- {brand}"),
        )
    } else if lower.contains("m4") {
        (38.0, format!("Apple Neural Engine (M4) -- {brand}"))
    } else if lower.contains("m3") {
        (18.0, format!("Apple Neural Engine (M3) -- {brand}"))
    } else if lower.contains("m2") {
        (15.8, format!("Apple Neural Engine (M2) -- {brand}"))
    } else if lower.contains("m1") {
        (11.0, format!("Apple Neural Engine (M1) -- {brand}"))
    } else {
        (
            PUBLISHED_FLOOR,
            format!("Apple Neural Engine (unrecognised chip) -- {brand}"),
        )
    }
}

#[cfg(all(test, target_os = "macos"))]
mod apple_tests {
    use super::estimate_apple_tops;

    #[test]
    fn newer_chips_are_not_reported_as_m1() {
        for brand in ["Apple M5", "Apple M5 Ultra", "Apple M6 Max", "Apple M9"] {
            let (tops, desc) = estimate_apple_tops(brand);
            assert!(tops >= 38.0, "{brand} reported only {tops} TOPS");
            assert!(desc.contains(brand));
        }
    }

    #[test]
    fn published_generations_keep_their_figures() {
        assert_eq!(estimate_apple_tops("Apple M1 Pro").0, 11.0);
        assert_eq!(estimate_apple_tops("Apple M2 Pro").0, 15.8);
        assert_eq!(estimate_apple_tops("Apple M3 Max").0, 18.0);
        assert_eq!(estimate_apple_tops("Apple M4").0, 38.0);
    }
}

// ── Linux: Intel OpenVINO NPU ──────────────────────────────────────────

#[cfg(target_os = "linux")]
fn detect_intel_linux() -> Option<NpuInfo> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    if !cpuinfo.contains("GenuineIntel") {
        return None;
    }

    let model_name = cpuinfo
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Intel CPU".to_string());

    let has_npu_device = std::path::Path::new("/dev/accel").exists()
        || std::fs::read_dir("/dev")
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .any(|e| e.file_name().to_string_lossy().starts_with("accel"))
            })
            .unwrap_or(false);

    if !has_npu_device {
        return None;
    }

    Some(NpuInfo {
        vendor: NpuVendor::Intel,
        tops: 11.0,
        max_precision: Precision::INT8,
        description: format!("Intel NPU -- {model_name}"),
    })
}

// ── Windows: Intel OpenVINO NPU ────────────────────────────────────────

#[cfg(target_os = "windows")]
fn detect_intel_windows() -> Option<NpuInfo> {
    let openvino_paths = [
        r"C:\Program Files (x86)\Intel\openvino\runtime\bin\intel64\Release\openvino.dll",
        r"C:\Program Files\Intel\openvino\runtime\bin\intel64\Release\openvino.dll",
    ];
    let has_openvino = openvino_paths
        .iter()
        .any(|p| std::path::Path::new(p).exists());
    if !has_openvino {
        return None;
    }

    Some(NpuInfo {
        vendor: NpuVendor::Intel,
        tops: 11.0,
        max_precision: Precision::INT8,
        description: "Intel NPU (OpenVINO detected)".to_string(),
    })
}

// ── Linux: Qualcomm Hexagon NPU ────────────────────────────────────────

/// Detect a Snapdragon SoC on Linux.
///
/// The device-tree model string is the reliable signal on ARM64 Linux;
/// `/proc/cpuinfo` reports the Qualcomm implementer ID (0x51) as a fallback.
/// Neither confirms that the QNN SDK is installed -- that is a build-time
/// concern handled in `build.rs`.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn detect_qualcomm_qualifier() -> Option<String> {
    // Android has no usable device-tree model and its `/proc/cpuinfo` is
    // deliberately sparse, so ask the property service for the SoC name.
    // Emulators report names like "ranchu"/"goldfish"; only accept values
    // that actually identify a Qualcomm part.
    #[cfg(target_os = "android")]
    for key in [
        "ro.soc.model",
        "ro.board.platform",
        "ro.hardware",
        "ro.product.board",
    ] {
        if let Ok(output) = std::process::Command::new("getprop").arg(key).output() {
            let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let lower = value.to_lowercase();
            if lower.contains("qualcomm")
                || lower.contains("snapdragon")
                || lower.contains("qcom")
                || lower.starts_with("sm")
                || lower.starts_with("msm")
                || lower.starts_with("qcm")
                || lower.starts_with("qcs")
            {
                return Some(value);
            }
        }
    }

    if let Ok(model) = std::fs::read_to_string("/proc/device-tree/model") {
        let model = model.trim_end_matches('\0').trim().to_string();
        if model.to_lowercase().contains("qualcomm") || model.to_lowercase().contains("snapdragon")
        {
            return Some(model);
        }
    }

    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let lower = cpuinfo.to_lowercase();
    // 0x51 is Qualcomm's ARM implementer ID.
    if lower.contains("qualcomm") || lower.contains("cpu implementer\t: 0x51") {
        return Some("Qualcomm Snapdragon".to_string());
    }
    None
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn detect_qualcomm_linux() -> Option<NpuInfo> {
    let model = detect_qualcomm_qualifier()?;
    Some(NpuInfo {
        vendor: NpuVendor::Qualcomm,
        // Conservative: Hexagon TOPS vary widely across Snapdragon generations
        // and are not discoverable without the SDK.
        tops: 45.0,
        max_precision: Precision::INT8,
        description: format!("Qualcomm Hexagon NPU -- {model}"),
    })
}

// ── Windows: Qualcomm Hexagon NPU ──────────────────────────────────────

/// Detect the QNN HTP backend on Windows on Snapdragon.
///
/// Presence of the HTP backend DLL is the practical signal: it ships with the
/// Snapdragon driver stack, so finding it means both the hardware and the
/// runtime are there.
#[cfg(target_os = "windows")]
fn detect_qualcomm_windows() -> Option<NpuInfo> {
    let candidates = [
        r"C:\Windows\System32\QnnHtp.dll",
        r"C:\Windows\System32\DriverStore\FileRepository\QnnHtp.dll",
    ];
    if !candidates.iter().any(|p| std::path::Path::new(p).exists()) {
        return None;
    }

    Some(NpuInfo {
        vendor: NpuVendor::Qualcomm,
        tops: 45.0,
        max_precision: Precision::INT8,
        description: "Qualcomm Hexagon NPU (QNN HTP backend detected)".to_string(),
    })
}
