# dlssnr-amd-lab

## Purpose

Proof-of-concept validation that RDNA3 GPUs can handle DLSS Neural Rendering–class workloads via AMD-native Vulkan compute, without relying on NVIDIA's NGX DLL or drivers.

## What This Is

Standalone Vulkan compute benchmarks measuring FP16 matrix multiplication throughput on Steam Machine (RADV NAVI33, SteamOS). The goal was to answer one question: **Is the hardware capable?**

The answer: yes. A simulated 52-layer neural frame sustains **10.8 ms/frame at 720p-equivalent** (scaled benchmark) — a factor of 2–3x away from a comfortable 3–5 ms budget; cooperative-matrix path (confirmed available) is the planned route to close the gap.

## What This Is Not

This does **not** reimplement DLSS NR and includes no NVIDIA binaries. It measures AMD GPU compute capability as a substrate for potential future open-model NR implementations.

## Measured Results

| Metric | Value |
|--------|-------|
| GPU | AMD Radeon Graphics (RADV NAVI33, 28 CUs) |
| Naive FP16 matmul (512×512) | 747 GFLOPS |
| Tiled FP16 (shared memory) | 1,459 GFLOPS |
| Simulated NR @ 720p-equiv (scaled) | 10.8 ms/frame |
| Cooperative matrix support | YES (f16×f16→f32 @ 16×16×16, subgroup scope) |

## Files

| File | Purpose |
|------|---------|
| `vulkan_test.cpp` + `matmul.comp` | Baseline fp16 matmul benchmark, correctness verified vs CPU |
| `tiled2.cpp` + `tiled2.comp` | Shared-memory tiled kernel |
| `coop_probe3.cpp` | Queries `VK_KHR_cooperative_matrix` configs exposed by the driver |
| `net_sim.cpp` + `net_sim.comp` | Multi-layer "neural frame" simulation, per-frame submission timing |
| `PROJECT_NOTES.md` | Track A validation notes, decisions, open problems |

## Building

Requires Vulkan headers and `glslangValidator` (SteamOS: `sudo pacman -S vulkan-headers glslang`).

    g++ -std=c++17 -O2 -o vulkan_test vulkan_test.cpp -lvulkan
    g++ -std=c++17 -O2 -o tiled2 tiled2.cpp -lvulkan
    g++ -std=c++17 -O2 -o coop_probe3 coop_probe3.cpp -lvulkan
    g++ -std=c++17 -O2 -o net_sim net_sim.cpp -lvulkan
    ./net_sim

## Methodology

1. **Correctness**: kernels verified against CPU reference (`[CHECK] MATCH`).
2. **Timing**: per-frame submission (warmup → timed batch → fence sync) captures realistic driver overhead.
3. **Tiles**: 16×16 shared-memory tiles matching the cooperative matrix config from the probe.

## Limitations

- `glslangValidator` currently rejects `GL_EXT_cooperative_matrix`; the tensor lane is measured via capability probe only.
- Synthetic weights; real NR models have different memory access profiles.
- Single-GPU (APU) measurements.

## License

MIT.

## Acknowledgments

- **Dagherbou / OptiScaler_DLSSNR** — Feature ID 18 wiring and Vulkan hook architecture.
- **RenoDX (clshortfuse)** — NR color-composition precedent.
- **Mesa / RADV team** — cooperative matrix exposure on consumer hardware.
- **OptiScaler upstream** — cross-upscaler bridging foundation.

Benchmark numbers from other RDNA3/4 cards welcome — open an Issue.
