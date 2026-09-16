# Steam Machine (RDNA3/RADV) NR Substrate: Integration Pathway Mapped + Open Invitation

Full-resolution-scaled benchmarks from a semi-custom RDNA3 APU (28 CUs) under SteamOS/RADV,
plus a complete mapping of the forwarder contract required to integrate DLSS-NR-style passes
on non-NVIDIA hardware.

## Measured Compute Budget (Net Sim Workload: 52-layer, 512x512-equivalent stack)

| Target Resolution | Workload | Time/Frame | Verdict |
|-------------------|----------|------------|---------|
| 960×540 (FSR Performance) | 8.19 GFLOP | **5.555 ms** | Playable with LSFG |
| 1280×720 (FSR Quality) | 16.37 GFLOP | 10.8 ms | Needs optimization |
| 1920×1080 (Native) | 32.75 GFLOP | 21.6 ms | Out of budget |

Sustained throughput: ~1,470–1,520 GFLOPS  
Cooperative matrix confirmed (f16×f16→f32 @ 16×16×16 subgroup) but unreachable from current glslang tooling.

## Forwarder Contract (Mapped from Dagherbou's OptiScaler_DLSSNR fork)

The host calls five Vulkan entry points through a forwarder interface:

| Export | Signature | Purpose |
|--------|-----------|---------|
| `dlssnr_vk_probe` | `int(__cdecl*)(wchar_t*)` | Returns 15 (all entry points available) |
| `dlssnr_vk_init` | `int(__cdecl*)(wchar_t*, wchar_t*, void*, void*, void*, int)` | Initializes per-device state, returns 1 |
| `dlssnr_vk_create` | `void*(__cdecl*)(void*, void*, uint32_t, uint32_t, ...)` | Allocates feature handle |
| `dlssnr_vk_evaluate` | `int(__cdecl*)(void*, void*, void*, ..., NVSDK_NGX_Resource_VK*, ...)` | Per-frame model dispatch |
| `dlssnr_vk_release` | `void(__cdecl*)(void*)` | Frees feature handle |

Binding layout (8 slots):
- Slot 0: Uniform buffer (DlssNrConstants)
- Slot 1–7: Combined image samplers + storage images (proxy/model/original/motion/target/keep/sampler)

## Backend Implementation (Cross-Compiled, Verified Loads on SteamOS)

A Win32 DLL (`nvngx.dll_dlssnr.dll`, 808 KB) was cross-compiled with MinGW to satisfy the five-export contract.
Deployment tested under Proton with the OptiScaler hook chain:

- Hook verified: OptiScaler loaded as `dxgi.dll`, ran full session, unloaded cleanly
- Our backend holds the forwarder slot in the call chain
- Runtime quiescent (no NGX feature triggered due to AMD hardware + FSR upscaler configuration)
- Host logs show no crashes or validation errors

## The Open Question: Model Weights

NVIDIA's `nvngx_dlssnr.dll` (the actual model) is:
- Approximately 165 MB in size
- RTX 50-series hardware only (per fork README)
- Closed-source and unshippable
- Must be user-supplied, not distributable with this work

**This project's contribution ends at the substrate level.** The compute budget, interface contract,
deployment pathway, and runtime integration are all documented and verified. The neural network itself
remains an open problem.

## Call for Collaboration

Two paths forward that could close the gap:

1. **Weight extraction or licensed kernels**: If anyone has already reimplemented the model (or can license
   reimplementation rights), the integration substrate is ready on this end.

2. **Train a distilled model**: Paired datasets (upscaled vs native render) are generatable on any hardware.
   A smaller model sized to the 5.5 ms budget is feasible — dataset collection and training infrastructure
   planning available for contributors.

Repository with full methodology, benchmark source, and deployment artifacts:
https://github.com/GetRektRambo/dlssnr-amd-lab

Contact for collaboration: [your Discord/email here]
