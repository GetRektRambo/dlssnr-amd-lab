// net_sim -- simulate one neural-rendering "frame":
// a chain of layer-sized matmuls, submitted per frame, measured end-to-end.
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#define CHECK(call, msg) \
    do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { fprintf(stderr, "[FAIL] %s (VkResult %d)\n", msg, (int)r_); exit(1); } } while (0)

static uint32_t findMemType(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return 0xFFFFFFFFu;
}

static inline uint16_t f32to16(float v) {
    union { float f; uint32_t u; } c; c.f = v;
    uint32_t sign = (c.u >> 16) & 0x8000u;
    int32_t exp = ((c.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (c.u >> 13) & 0x3FFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | (uint32_t(exp) << 10) | mant);
    return (uint16_t)(sign | (uint32_t(exp) << 10) | mant);
}
static inline float h16to32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) return 0.0f;
    union { float f; uint32_t u; } c;
    c.u = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    return c.f;
}

// Same proven kernel from tiled2 (push-constant DIM, shared-memory tiling)
static const char* g_shader_src = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(push_constant) uniform Push { uint DIM; } pc;
layout(set=0,binding=0) readonly buffer A { uint a[]; };
layout(set=0,binding=1) readonly buffer B { uint b[]; };
layout(set=0,binding=2) writeonly buffer C { float c[]; };
shared float tileA[16][16];
shared float tileB[16][16];
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint lrow = gl_LocalInvocationID.y;
    uint lcol = gl_LocalInvocationID.x;
    uint DIM = pc.DIM;
    if (row >= DIM || col >= DIM) return;
    float acc = 0.0;
    for (uint t = 0u; t < DIM; t += 16u) {
        tileA[lrow][lcol] = unpackHalf2x16(a[row * DIM + t + lcol]).x;
        tileB[lrow][lcol] = unpackHalf2x16(b[(t + lrow) * DIM + col]).x;
        barrier();
        for (uint k = 0u; k < 16u; k++)
            acc += tileA[lrow][k] * tileB[k][lcol];
        barrier();
    }
    c[row * DIM + col] = acc;
}
)";

// One simulated "network": layer size x layer count (largest last so the
// correctness check matches a 512-wide final layer)
struct LayerSpec { uint32_t dim; uint32_t count; };
static const LayerSpec kLayers[] = { {128, 16}, {256, 24}, {512, 12} };

int main() {
    const uint32_t MAXDIM = 512;
    const uint32_t FRAMES = 100;

    printf("[Net Sim] Neural-frame budget simulation\n\n");

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

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qIdx = 0;
    for (uint32_t i = 0; i < qn; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qIdx = i; break; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo dq{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dq.queueFamilyIndex = qIdx; dq.queueCount = 1; dq.pQueuePriorities = &prio;
    VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dc.queueCreateInfoCount = 1; dc.pQueueCreateInfos = &dq;
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dc, nullptr, &dev), "create device");
    VkQueue queue;
    vkGetDeviceQueue(dev, qIdx, 0, &queue);

    // Write + compile shader
    FILE* fs = fopen("net_sim.comp", "wb");
    fwrite(g_shader_src, 1, strlen(g_shader_src), fs);
    fclose(fs);
    if (system("glslangValidator -V net_sim.comp -o net_sim.spv") != 0) {
        fprintf(stderr, "[FAIL] shader compilation failed\n");
        return 1;
    }
    fs = fopen("net_sim.spv", "rb");
    fseek(fs, 0, SEEK_END); long sz = ftell(fs); fseek(fs, 0, SEEK_SET);
    std::vector<char> code(sz);
    if (fread(code.data(), 1, sz, fs) != (size_t)sz) return 1;
    fclose(fs);

    VkShaderModuleCreateInfo smc{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smc.codeSize = code.size(); smc.pCode = (uint32_t*)code.data();
    VkShaderModule module;
    CHECK(vkCreateShaderModule(dev, &smc, nullptr, &module), "shader module");

    // Max-size buffers; small layers just use a subregion
    VkDeviceSize bufSize = (VkDeviceSize)MAXDIM * MAXDIM * 4;
    VkBuffer bufs[3]; VkDeviceMemory mems[3];
    for (int i = 0; i < 3; i++) {
        VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bc.size = bufSize; bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        CHECK(vkCreateBuffer(dev, &bc, nullptr, &bufs[i]), "create buffer");
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, bufs[i], &mr);
        VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ma.allocationSize = mr.size;
        ma.memoryTypeIndex = findMemType(pd, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        CHECK(vkAllocateMemory(dev, &ma, nullptr, &mems[i]), "alloc");
        CHECK(vkBindBufferMemory(dev, bufs[i], mems[i], 0), "bind");
    }

    uint32_t* mapA = nullptr; uint32_t* mapB = nullptr; float* mapC = nullptr;
    vkMapMemory(dev, mems[0], 0, bufSize, 0, (void**)&mapA);
    vkMapMemory(dev, mems[1], 0, bufSize, 0, (void**)&mapB);
    vkMapMemory(dev, mems[2], 0, bufSize, 0, (void**)&mapC);
    srand(1234);
    for (uint32_t i = 0; i < MAXDIM * MAXDIM; i++) {
        mapA[i] = (uint32_t)f32to16(((rand() % 200) - 100) / 100.0f);
        mapB[i] = (uint32_t)f32to16(((rand() % 200) - 100) / 100.0f);
    }

    // Pipeline plumbing (descriptor set + push-constant layout)
    VkDescriptorSetLayoutBinding lb[3] = {};
    for (int i = 0; i < 3; i++) { lb[i].binding = i; lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; lb[i].descriptorCount = 1; lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 3; dl.pBindings = lb;
    VkDescriptorSetLayout setLayout;
    CHECK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &setLayout), "desc layout");

    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo plc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plc.setLayoutCount = 1; plc.pSetLayouts = &setLayout;
    plc.pushConstantRangeCount = 1; plc.pPushConstantRanges = &push;
    VkPipelineLayout pipeLayout;
    CHECK(vkCreatePipelineLayout(dev, &plc, nullptr, &pipeLayout), "pipeline layout");

    VkComputePipelineCreateInfo pc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pc.layout = pipeLayout;
    pc.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pc.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.stage.module = module; pc.stage.pName = "main";
    VkPipeline pipe;
    CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pc, nullptr, &pipe), "pipeline");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpc.maxSets = 1; dpc.poolSizeCount = 1; dpc.pPoolSizes = &ps;
    VkDescriptorPool pool;
    CHECK(vkCreateDescriptorPool(dev, &dpc, nullptr, &pool), "desc pool");
    VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool = pool; da.descriptorSetCount = 1; da.pSetLayouts = &setLayout;
    VkDescriptorSet set;
    CHECK(vkAllocateDescriptorSets(dev, &da, &set), "desc set");

    static VkDescriptorBufferInfo binfos[3];
    VkWriteDescriptorSet w[3] = {};
    for (int i = 0; i < 3; i++) {
        binfos[i] = {bufs[i], 0, bufSize};
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = set; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &binfos[i];
    }
    vkUpdateDescriptorSets(dev, 3, w, 0, nullptr);

    VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpc.queueFamilyIndex = qIdx;
    VkCommandPool cp;
    CHECK(vkCreateCommandPool(dev, &cpc, nullptr, &cp), "cmd pool");
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = cp; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    VkCommandBuffer cb;
    CHECK(vkAllocateCommandBuffers(dev, &ca, &cb), "cmd buffer");

    VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(dev, &fc, nullptr, &fence), "fence");

    // One "frame" = all layer dispatches, submitted individually per frame
    auto recordFrame = [&]() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK(vkBeginCommandBuffer(cb, &bi), "begin");
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &set, 0, nullptr);
        for (const auto& L : kLayers) {
            vkCmdPushConstants(cb, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(L.dim), &L.dim);
            for (uint32_t i = 0; i < L.count; i++)
                vkCmdDispatch(cb, L.dim / 16, L.dim / 16, 1);
        }
        CHECK(vkEndCommandBuffer(cb), "end");
    };
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;

    auto runFrame = [&]() {
        recordFrame();
        CHECK(vkQueueSubmit(queue, 1, &si, fence), "submit frame");
        CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "wait frame");
        CHECK(vkResetFences(dev, 1, &fence), "reset fence");
    };

    for (int i = 0; i < 5; i++) runFrame();  // warmup

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t f = 0; f < FRAMES; f++) runFrame();
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    // Correctness: final layer is 512 wide
    float expect = 0.0f;
    for (uint32_t k = 0; k < 512; k++)
        expect += h16to32((uint16_t)(mapA[k] & 0xFFFF)) * h16to32((uint16_t)(mapB[k * 512] & 0xFFFF));
    printf("[CHECK] c[0] GPU=%.2f vs CPU=%.2f -> %s\n\n", mapC[0], expect,
           (fabsf(mapC[0] - expect) < 4.0f) ? "MATCH" : "MISMATCH");

    // Frame accounting
    double totalFlopPerFrame = 0.0;
    uint32_t totalDispatches = 0;
    for (const auto& L : kLayers) {
        totalFlopPerFrame += 2.0 * (double)L.dim * L.dim * L.dim * L.count;
        totalDispatches += L.count;
    }
    double msPerFrame = (secs / FRAMES) * 1000.0;
    double gflops = (totalFlopPerFrame / (secs / FRAMES)) / 1e9;

    printf("=== NET SIM RESULTS ===\n");
    printf("Network: %u layers, %u dispatches per frame\n",
           16u + 24u + 12u, totalDispatches);
    printf("Workload: %.2f GFLOP per frame\n", totalFlopPerFrame / 1e9);
    printf("Frames simulated: %u\n\n", FRAMES);
    printf("Time per frame:     %.3f ms\n", msPerFrame);
    printf("Effective rate:     %.1f GFLOPS sustained\n\n", gflops);

    printf("Budget verdict (60fps = 16.6 ms total, NR share ~3-5 ms):\n");
    if (msPerFrame < 3.0)      printf("  [EXCELLENT] Fits comfortably inside a realistic NR slice.\n");
    else if (msPerFrame < 5.0) printf("  [GOOD] Fits the generous end of an NR budget.\n");
    else if (msPerFrame < 8.0) printf("  [MARGINAL] Workable at reduced resolution or lower fps.\n");
    else                       printf("  [CAUTION] Exceeds a playable NR slice as-is.\n");

    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, cp, nullptr);
    vkDestroyDescriptorPool(dev, pool, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(dev, setLayout, nullptr);
    vkDestroyShaderModule(dev, module, nullptr);
    vkUnmapMemory(dev, mems[0]); vkUnmapMemory(dev, mems[1]); vkUnmapMemory(dev, mems[2]);
    for (int i = 0; i < 3; i++) { vkDestroyBuffer(dev, bufs[i], nullptr); vkFreeMemory(dev, mems[i], nullptr); }
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
