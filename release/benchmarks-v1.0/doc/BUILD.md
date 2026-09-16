# Building the NR Substrate Benchmarks

Requires: Linux, Vulkan headers, g++ (or clang++), RADV driver recommended.

    g++ -std=c++17 -O2 -o net_sim net_sim.cpp -lvulkan
    g++ -std=c++17 -O2 -o tiled2 tiled2.cpp -lvulkan
    g++ -std=c++17 -O2 -o coop_probe3 coop_probe3.cpp -lvulkan
    g++ -std=c++17 -O2 -o vulkan_test vulkan_test.cpp -lvulkan

Run order: vulkan_test (sanity), tiled2 (throughput), coop_probe3 (matrix ops), net_sim (NR cost).

net_sim takes width/height arguments:
    ./net_sim 1280 720    # FSR Quality at 1080p output
    ./net_sim 960 540     # FSR Performance
    ./net_sim 1920 1080   # Native
