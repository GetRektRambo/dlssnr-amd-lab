// Probe v3: correct proc-address usage (physical-device fn comes from INSTANCE)
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call, msg) \
    do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { fprintf(stderr, "[FAIL] %s (VkResult %d)\n", msg, (int)r_); exit(1); } } while (0)

static const char* ct2s(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
        case VK_COMPONENT_TYPE_SINT8_KHR:   return "i8";
        case VK_COMPONENT_TYPE_UINT8_KHR:   return "u8";
        case VK_COMPONENT_TYPE_SINT32_KHR:  return "i32";
        default: return "?";
    }
}
static const char* sc2s(VkScopeKHR s) {
    switch (s) {
        case VK_SCOPE_DEVICE_KHR:     return "device";
        case VK_SCOPE_WORKGROUP_KHR:  return "workgroup";
        case VK_SCOPE_SUBGROUP_KHR:   return "subgroup";
        default: return "?";
    }
}

int main() {
    printf("[Cooperative Matrix Probe v3]\n\n");

    // Plain instance -- no extensions listed (they're device-level)
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, nullptr, &inst), "create instance");

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> gpus(n);
    CHECK(vkEnumeratePhysicalDevices(inst, &n, gpus.data()), "enumerate GPUs");
    VkPhysicalDevice pd = gpus[0];

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    printf("[INFO] GPU: %s\n", props.deviceName);

    // THE FIX: physical-device functions are fetched from the INSTANCE, not the device
    auto getProps = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!getProps) {
        fprintf(stderr, "[FAIL] query fn not in instance dispatch table\n");
        return 1;
    }

    uint32_t pn = 0;
    getProps(pd, &pn, nullptr);
    if (pn == 0) {
        printf("Query worked, but 0 configurations reported.\n");
        return 1;
    }
    std::vector<VkCooperativeMatrixPropertiesKHR> ps(pn);
    for (auto& p : ps) p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    getProps(pd, &pn, ps.data());

    printf("\nSupported cooperative matrix configurations:\n");
    printf("%-6s %-6s %-6s  M x N x K   scope\n", "A", "B", "result");
    printf("-----------------------------------------------\n");
    for (const auto& p : ps) {
        printf("%-6s %-6s %-6s  %2u x %2u x %2u   %s%s\n",
               ct2s(p.AType), ct2s(p.BType), ct2s(p.CType),
               p.MSize, p.NSize, p.KSize, sc2s(p.scope),
               p.saturatingAccumulation ? " (sat)" : "");
    }
    return 0;
}
