#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# Ensure dependencies are present
if ! command -v x86_64-w64-mingw32-g++ &> /dev/null; then
    echo "[ERROR] MinGW cross-compiler not found. Run in crossdev distrobox."
    exit 1
fi

# Create package dir
PKG_DIR="build/dlssnr-amd-pack-v1.0"
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR"

# Build the backend DLL
x86_64-w64-mingw32-g++ -std=c++17 -shared -O2 \
    -I include \
    -o "$PKG_DIR/nvngx.dll_dlssnr.dll" src/amd_vk_nr_backend.cpp \
    -L include/vulkan -l:libvulkan-1.a \
    -static -static-libgcc -static-libstdc++

# Create dummy snippet
touch "$PKG_DIR/nvngx_dlssnr.dll"

# Add docs
cp INSTALL.txt "$PKG_DIR/" 2>/dev/null || echo "Install instructions" > "$PKG_DIR/INSTALL.txt"
cp COMMUNITY_POST.md "$PKG_DIR/README.md"

# Verify exports
x86_64-w64-mingw32-objdump -p "$PKG_DIR/nvngx.dll_dlssnr.dll" | grep "dlssnr_vk" || (echo "[FAIL] Missing exports"; exit 1)

echo "[OK] Package ready at $PKG_DIR"
ls -la "$PKG_DIR/"
