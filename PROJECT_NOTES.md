# DLSS-NR-on-AMD Track A: Hardware Validation Notes

## Measured (Steam Machine, RADV NAVI33, SteamOS)
- Naive fp16 matmul:        747 GFLOPS  (512x512, scalar kernel)
- Tiled fp16 matmul:        1459 GFLOPS (512x512, 16x16 shared-memory tiles)
- Coop-matrix configs:      f16xf16->f16/f32 @ 16x16x16 (subgroup) -- CONFIRMED
                            u8/i8 -> i32 configs also available
- GLSL coop-matrix path:    NOT available in glslang (ext rejected); future work

## Decisions
- Track A verdict: PASS (correctness + throughput + tensor lane exists)
- Production kernel for now: tiled shared-memory shader (net_sim.comp / tiled2.comp)
- Integration deferred until frame-budget sim passes (pending net_sim results)

## Known open problems
- OptiScaler is a Windows DLL under Proton; Vulkan backend is host-side Linux.
  Bridge architecture needed (likely out-of-process + shared memory).
- NR model weights remain the long-term blocker (proprietary inside NGX DLLs).

---

## Session 2026-09-15 (c): Resolution-scaled benchmarks + landscape

### Scaling fix applied to net_sim
- net_sim now takes WIDTH HEIGHT args; workload scales by pixel count
  (720p = 4 stacks, 1080p = 8 stacks of the 52-layer 512x512 net)
- Full ladder results in results/ (compare net_sim_720 vs net_sim_1080)

### Landscape (verified this session)
- danielblnc/DLSS-NR-on-AMD: working NR reimpl for AMD, HIP kernels
  (gfx1100/1101/1102/1201), Windows-only. Alpha 0.3.0 adds pre-upscaling
  mode: NR runs at INTERNAL resolution before FSR — validates our
  720p-equivalent scaling assumption for Steam Machine.
- FSR Redstone (AMD official): ML NR features RDNA4-only; RDNA3+ gets
  analytical fallbacks. SDK is useful reference for AMD-native NN runtime
  structure. No RADV/Linux path stated.

### Direction confirmed
- Hybrid plan: fork danielblnc's runtime skeleton, replace HIP kernels with
  lab-validated SPIR-V compute, deploy via decky-dlss5lab pipeline.
- First test target: Marvel's Spider-Man Remastered (existing OptiScaler
  Feature ID 18 routing + logs already validated against this game).
- Fatbin compat open question: need exact gfx arch (1102 vs 1103/Phoenix);
  see results/gpu_ident_NAVI33.txt. SPIR-V port removes this dependency.

### Raw results this session
- see results/*.txt (vulkan_test, tiled2, coop_probe3, net_sim 720/1080)
