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
