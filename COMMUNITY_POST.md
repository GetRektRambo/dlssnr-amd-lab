# Steam Machine (RDNA3/RADV) NR Substrate: Measured Benchmarks + Roadmap

Full-resolution-scaled benchmarks from a semi-custom RDNA3 APU (28 CUs) under SteamOS/RADV.
NR-class workload (52-layer stack, per-frame submission, correctness verified vs CPU):

| Target | Workload | Time/Frame | Verdict |
|--------|----------|------------|---------|
| 960×540 (FSR Performance) | 8.19 GFLOP | **5.555 ms** | **Playable with LSFG** |
| 1280×720 (FSR Quality) | 16.37 GFLOP | 10.8 ms | 2–3× optimization needed |
| 1920×1080 (Native) | 32.75 GFLOP | 21.6 ms | Out of budget |

Sustained throughput: ~1,470–1,520 GFLOPS  
Cooperative matrix confirmed (f16×f16→f32 @ 16×16×16 subgroup) but unreachable from current glslang tooling.

## Framing: NR + LSFG Pipeline

Paired with LSFG frame generation (30 rendered → 60 displayed), the per-frame budget doubles to ~33 ms. At 540p internal resolution, the combined stack estimates to ~18–22 ms, leaving 11–15 ms headroom:

- Game render (540p): ~10–12 ms
- NR pass (measured): 5.555 ms
- FSR + LSFG overhead: ~3–4 ms

## Next Steps

Hybrid port underway: fork danielblnc's runtime skeleton, replace HIP kernels with SPIR-V compute shaders validated on this benchmark suite, deploy via decky-dlss5lab pipeline. First test target: *Marvel's Spider-Man Remastered* (Feature ID 18 routing already validated against this game).

Open calls:
- coop_probe3 results from other AMD owners (RX 7800 XT/7900/8000/9000 series) to map the hardware landscape
- Input on motion-vector interception under DXVK override (high-complexity subsystem)

Repository with full methodology and raw data: https://github.com/GetRektRambo/dlssnr-amd-lab
