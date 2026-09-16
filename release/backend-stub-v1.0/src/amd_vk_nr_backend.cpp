// ============================================================================
// amd_vk_nr_backend.cpp — Stage 1 stub: native AMD NR backend
//
// Replaces the nvngx_dlssnr.dll + forwarder pair with a native Vulkan
// implementation of the same five-entry-point contract.
//
//   Stage 1 (this file): plumbing only. Random/fixed weights, correct
//   contracts, dispatch through our validated net_sim kernel stack later.
//
// Build target: Linux/SteamOS (RADV). On Linux there is no __cdecl.
// ============================================================================

#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdarg>

#if defined(_WIN32)
#define NR_CALL __cdecl
#define NR_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define NR_CALL
#define NR_EXPORT __attribute__((visibility("default")))
#endif

// ---------------------------------------------------------------------------
// Canonical NGX resource layout — use the real SDK header, not a mirror.
// The header is declarative only (structs/enums, no driver linkage).
// ---------------------------------------------------------------------------
#include "nvsdk_ngx_vk.h"  // vendored beside this file

// ---------------------------------------------------------------------------
// Per-feature state (one per created feature handle)
// ---------------------------------------------------------------------------
struct FeatureHandle
{
    uint32_t workWidth = 0;
    uint32_t workHeight = 0;
    float intensity = 1.0f;
    int style = 0;
    float localStructure = 1.0f;
    float localTone = 1.0f;
    float skinStructure = 1.0f;
    int autoMask = 0;

    // Populated during Stage 2:
    // VkPipeline pipeline; VkPipelineLayout pipelineLayout;
    // VkDescriptorPool descriptorPool; VkDescriptorSetLayout setLayout;
};

// ---------------------------------------------------------------------------
// Global device state (set by init, cleared by a release of the last feature)
// ---------------------------------------------------------------------------
namespace
{
VkInstance g_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_physicalDevice = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
float g_timestampPeriod = 0.0f;

unsigned long long g_frames = 0;

void nrLog(const char* fmt, ...)
{
    // Stage 1: plain stdout. Wire into OptiScaler's LOG_INFO in Stage 3.
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}
} // namespace

// ============================================================================
// The five-contract exports, signatures matching the forwarder typedefs
// ============================================================================

extern "C" {
// exports are marked individually below
// Probe: four bits, one per entry point. We are the model: everything answers.
NR_EXPORT int NR_CALL dlssnr_vk_probe(const wchar_t* snippetPathIgnored)
{
    (void) snippetPathIgnored;

    nrLog("[AMD-NR] probe: native backend present, struct sizes: resource=%zu imageview=%zu",
          sizeof(NVSDK_NGX_Resource_VK), sizeof(NVSDK_NGX_ImageViewInfo_VK));

    return 15; // all four entry points reachable
}

// Init: store the device, verify timestamp support for later in-game timing.
NR_EXPORT int NR_CALL dlssnr_vk_init(const wchar_t* pathA, const wchar_t* pathB, void* instance,
                           void* physicalDevice, void* device, int flags)
{
    (void) pathA; (void) pathB; (void) flags;

    g_instance = (VkInstance) instance;
    g_physicalDevice = (VkPhysicalDevice) physicalDevice;
    g_device = (VkDevice) device;

    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(g_physicalDevice, &props);
    g_timestampPeriod = props.limits.timestampPeriod;

    nrLog("[AMD-NR] init: device %04x:%04x, API %u, timestampPeriod %.1f ns",
          props.vendorID, props.deviceID, props.apiVersion, (double) g_timestampPeriod);

    return 1; // success
}

// Create: allocate feature state for one working size.
NR_EXPORT void* NR_CALL dlssnr_vk_create(void* cmdBuffer, void* capabilityParams, unsigned int workWidth,
                               unsigned int workHeight, int preset, float intensity, int style,
                               float localStructure, float localTone, float skinStructure,
                               int autoMask, int unused)
{
    (void) cmdBuffer; (void) capabilityParams; (void) preset; (void) unused;

    if (g_device == VK_NULL_HANDLE)
        return nullptr;

    FeatureHandle* feat = new FeatureHandle {};
    feat->workWidth = workWidth;
    feat->workHeight = workHeight;
    feat->intensity = intensity;
    feat->style = style;
    feat->localStructure = localStructure;
    feat->localTone = localTone;
    feat->skinStructure = skinStructure;
    feat->autoMask = autoMask;

    nrLog("[AMD-NR] create: feature up at %ux%u", workWidth, workHeight);

    // Stage 2 here: build the descriptor pool / pipeline / SPIR-V modules,
    // cached against (workWidth, workHeight).

    return feat;
}

// Evaluate: the per-frame hot path.
NR_EXPORT int NR_CALL dlssnr_vk_evaluate(void* cmdBuffer, void* feature, void* params,
                               void* colour, void* depth, void* motion, void* output,
                               unsigned int workWidth, unsigned int workHeight,
                               unsigned int guideWidth, unsigned int guideHeight,
                               int depthInverted, int reset, float intensity, int style,
                               float localStructure, float localTone, float skinStructure,
                               int autoMask, float unusedA, float unusedB)
{
    (void) params; (void) guideWidth; (void) guideHeight; (void) depthInverted;
    (void) unusedA; (void) unusedB;

    FeatureHandle* feat = static_cast<FeatureHandle*>(feature);
    VkCommandBuffer cmd = (VkCommandBuffer) cmdBuffer;

    if (feat == nullptr || cmd == VK_NULL_HANDLE)
        return 0;

    // Update the per-frame knobs; the host may retune them without recreating.
    feat->intensity = intensity;
    feat->style = style;
    feat->localStructure = localStructure;
    feat->localTone = localTone;
    feat->skinStructure = skinStructure;
    feat->autoMask = autoMask;

    NVSDK_NGX_Resource_VK* colourRes = static_cast<NVSDK_NGX_Resource_VK*>(colour);
    NVSDK_NGX_Resource_VK* outputRes = static_cast<NVSDK_NGX_Resource_VK*>(output);

    if (colourRes == nullptr || outputRes == nullptr)
        return 0;

    const VkImageView srcView = colourRes->Resource.ImageViewInfo.ImageView;
    const VkImageView dstView = outputRes->Resource.ImageViewInfo.ImageView;

    if (srcView == VK_NULL_HANDLE || dstView == VK_NULL_HANDLE)
        return 0;

    // ---- Stage 1: record a dispatch-shaped no-op so the command buffer is
    // touched and the timestamp pair in the host brackets *something*.
    //
    // A pipeline barrier keeps the pass honest about ordering without
    // needing a pipeline yet; the real 52-layer stack replaces this block.
    VkMemoryBarrier marker {};
    marker.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    marker.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    marker.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &marker, 0, nullptr,
                         0, nullptr);

    if (reset)
        g_frames = 0;
    else
        ++g_frames;

    if (g_frames == 1)
        nrLog("[AMD-NR] evaluate: first frame, %ux%u, source view %p, dest view %p",
              workWidth, workHeight, (void*) (uintptr_t) srcView, (void*) (uintptr_t) dstView);
    else if (g_frames == 100 || g_frames % 1000 == 0)
        nrLog("[AMD-NR] evaluate: frame %llu", g_frames);

    return 1; // success
}

// Release: drop feature state.
NR_EXPORT void NR_CALL dlssnr_vk_release(void* feature)
{
    FeatureHandle* feat = static_cast<FeatureHandle*>(feature);

    if (feat == nullptr)
        return;

    // Stage 2 here: destroy the pipeline/descriptor caches.
    delete feat;
}

} // extern "C"
