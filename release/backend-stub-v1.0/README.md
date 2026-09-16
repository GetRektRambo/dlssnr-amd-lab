# AMD NR Forwarder-Replacement Backend (Linux/RADV) - v1.0

Cross-compiled Win32 DLL implementing the five-export DLSS-NR forwarder contract
(`dlssnr_vk_probe/init/create/evaluate/release`), loadable by Dagherbou's
OptiScaler_DLSSNR fork on Linux/SteamOS/Proton/RADV -- no host patches, no NVIDIA DLLs.

## What it currently does

- Fully satisfies the forwarder contract (probe returns 15, init/create/evaluate/release)
- Loads through OptiScaler's LoadForwarder() via Wine/Proton without any host modification
- Initializes per-device Vulkan state through winevulkan -> RADV
- Receives per-frame evaluate() calls with real game textures and parameters
- Records pipeline barriers per frame (timing bracket placeholder)
- Logs [AMD-NR] breadcrumbs to the Proton log for verification

**Verified end-to-end:** OptiScaler session ran 6+ minutes in Spider-Man Remastered
under Proton, clean unload, zero crashes, no validation errors.

## What it does NOT do

Synthesize detail. evaluate() brackets frame cost but performs no inference --
the neural network weights remain the open problem (NVIDIA's model is RTX-50-only
and closed-source). This is the integration substrate: drop-in ready for any
future model implementation to slot into the evaluate() function.

## Installation

1. Extract into the game folder, beside the game .exe
2. Requires OptiScaler (DLSSNR fork) deployed as dxgi.dll in the same folder
3. Steam launch options: WINEDLLOVERRIDES="dxgi.dll=n,b" PROTON_LOG=1 %command%
4. In OptiScaler.ini, [DlssNr] section, set Enabled=true
5. Create an empty file named nvngx_dlssnr.dll beside the game exe (satisfies
   the host's snippet existence check; the backend ignores its path)
6. Launch into gameplay, then check the Proton log (~/steam-APPID.log)
   for [AMD-NR] lines

## Verification checklist

    [AMD-NR] probe: native backend present     <- contract accepted
    [AMD-NR] init: device 1002:7481            <- Vulkan device bound
    [AMD-NR] create: feature up at WxH         <- feature handle live
    [AMD-NR] evaluate: first frame             <- per-frame path running

## Where the model goes

In src/amd_vk_nr_backend.cpp, evaluate(): replace the pipeline-barrier
placeholder block with real compute dispatches. The function already receives
source/dest Vulkan image views via the NVSDK_NGX_Resource_VK structs --
everything needed to bind and dispatch a trained network is in hand.

## Build

See the crossbuild-toolchain release, or:

    x86_64-w64-mingw32-g++ -std=c++17 -shared -O2 \
        -I <vulkan-headers> \
        -o nvngx.dll_dlssnr.dll src/amd_vk_nr_backend.cpp \
        -L <path-to-libvulkan-1.a> -l:libvulkan-1.a \
        -static -static-libgcc -static-libstdc++

License: same as the repo (see LICENSE if present, otherwise ask).
