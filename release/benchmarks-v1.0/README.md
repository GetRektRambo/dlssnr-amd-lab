# NR Substrate Benchmarks (RDNA3 / SteamOS / RADV) [Final]

Resolution-scaled NR-class compute benchmarks for semi-custom RDNA3 APUs (28 CU, RADV, SteamOS).
First published performance envelope for neural-rendering-class workloads on console-class AMD silicon.

## What it does

- **vulkan_test**: baseline compute sanity check (naive FP16)
- **tiled2**: tiled-matrix FP16 throughput (vs naive)
- **coop_probe3**: cooperative matrix / WMMA configuration probe
- **net_sim**: simulated 52-layer NR frame cost at any resolution (args: width height)
- Correctness verified against CPU reference at every step
- Full source + prebuilt binaries + raw results included

## Headline results (Steam Machine, semi-custom RDNA3 APU, 28 CU, deviceID 0x7481)

| Resolution | Workload | Time/Frame | Verdict |
|---|---|---|---|
| 960x540 (FSR Performance) | 8.19 GFLOP | 5.555 ms | Playable with LSFG |
| 1280x720 (FSR Quality) | 16.37 GFLOP | 10.8 ms | Needs optimization |
| 1920x1080 (native) | 32.75 GFLOP | 21.6 ms | Out of budget |

Sustained throughput: ~1,470-1,520 GFLOPS FP16.
Cooperative matrix: confirmed f16xf16->f32 @ 16x16x16 subgroup, but unreachable
from current glslang tooling (documented toolchain gap).

## What it's for

Determining whether your AMD hardware can afford an NR pass in your frame budget.
Comparing your own GPU's NR-class throughput against a known reference.
Validating NR cost claims against a correct, resolution-scaled workload.

## Attribution

Built as part of the dlssnr-amd-lab project: measuring the substrate for
DLSS 5 Neural Rendering on Linux/AMD. See the main repo for the full story.
