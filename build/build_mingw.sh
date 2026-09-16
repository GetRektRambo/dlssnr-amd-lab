#!/bin/bash
# Cross-build the NR native backend as the forwarder-replacement DLL.
# Run inside the crossdev distrobox (needs mingw-w64-gcc + dlltool).
set -e
cd "$(dirname "$0")/.."
x86_64-w64-mingw32-dlltool -d build/vulkan.def -l build/libvulkan-1.a
# NB: source MUST precede the library on the link line — ld scans archives
# in order and pulls nothing if the object hasn't been seen yet.
x86_64-w64-mingw32-g++ -std=c++17 -shared -O2 \
    -I include \
    -o build/nvngx.dll_dlssnr.dll src/amd_vk_nr_backend.cpp \
    -L build -l:libvulkan-1.a \
    -static -static-libgcc -static-libstdc++
file build/nvngx.dll_dlssnr.dll
