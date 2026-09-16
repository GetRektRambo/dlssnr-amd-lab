# SteamOS MinGW/Vulkan Cross-Build Kit - v1.0

Reproducible recipe for building Win32 Vulkan-consuming DLLs from an
unmodified SteamOS machine (or any Linux box).

## The three problems it solves

1. **No MinGW on host** -- SteamOS is read-only and ships no cross-toolchain.
   Solution: distrobox container (Arch), installed to ~/.local, no sudo on host.

2. **No Vulkan import library for the cross target** -- mingw-w64-gcc on Arch
   ships no vulkan-1 import lib. Solution: mint your own with dlltool from a
   .def file listing only the symbols you call.

3. **Vulkan header tree placement** -- vulkan_core.h unconditionally includes
   vk_video/*.h, which are a SIBLING directory of vulkan/, not inside it.
   Solution: vendor the complete tree (vulkan/ + vk_video/) under one include root.

## Contents

- vulkan.def            -- two-symbol def (vkGetPhysicalDeviceProperties, vkCmdPipelineBarrier)
- libvulkan-1.a         -- dlltool-minted import library
- build_mingw.sh        -- one-command build of the backend stub
- build_release.sh      -- packaging script
- headers are in the repo's include/ tree (not duplicated here)

## Usage

1. Install distrobox, create the container:
       ~/.local/bin/distrobox create -n crossdev -i archlinux:latest
       ~/.local/bin/distrobox enter crossdev
2. Inside: sudo pacman -S --noconfirm mingw-w64-gcc
3. Clone this repo (or enter from a host with it mounted), run:
       bash build/build_mingw.sh

## Verification

    file build/nvngx.dll_dlssnr.dll
    -> PE32+ executable (DLL) (console) x86-64
    x86_64-w64-mingw32-objdump -p <dll> | grep dlssnr_vk   # five exports
    x86_64-w64-mingw32-objdump -p <dll> | grep vulkan-1     # import table

## Notes

- vulkan-1.dll resolves via winevulkan in any Proton prefix at load time.
- -static runtime libs mean zero external dependencies for consumers.
- The def-file technique extends to any additional Vulkan functions:
  add the symbol to vulkan.def, re-run dlltool, rebuild.
