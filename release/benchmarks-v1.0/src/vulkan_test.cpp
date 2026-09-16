// Real Vulkan fp16 matmul benchmark for Steam Machine RDNA3
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>

#define CHECK(call, msg) \
    do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { \
        fprintf(stderr, "[FAIL] %s (VkResult %d)\n", msg, (int)r_); exit(1); } } while (0)

static const uint32_t DIM = 512;
static const uint32_t RUNS = 200;   // matmuls per timed batch

static uint32_t findMemType(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return 0xFFFFFFFF;
}

int main() {
    printf("[DLSSNR compute test] Steam Machine RDNA3 fp16 benchmark\n\n");

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, nullptr, &inst), "create instance");

    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(inst, &gpuCount, nullptr);
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    CHECK(vkEnumeratePhysicalDevices(inst, &gpuCount, gpus.data()), "enumerate GPUs");
    VkPhysicalDevice pd = gpus[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    printf("[INFO] GPU: %s\n", props.deviceName);
    printf("[INFO] Vulkan driver version: %u.%u.%u\n",
           VK_VERSION_MAJOR(props.driverVersion), VK_VERSION_MINOR(props.driverVersion),
           VK_VERSION_PATCH(props.driverVersion));

    // Does the driver expose cooperative matrices (the fast tensor path)?
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &coop;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    printf("[INFO] Cooperative matrix support: %s\n",
           coop.cooperativeMatrix ? "YES (fast tensor path available)" : "NO (scalar path only)");

    // Compute queue
    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, qf.data());
    uint32_t qIdx = 0;
    for (uint32_t i = 0; i < qCount; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qIdx = i; break; }

    // Device with fp16 enabled (core feature in Vulkan 1.2)
    VkPhysicalDeviceShaderFloat16Int8Features f16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    f16.shaderFloat16 = VK_TRUE;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dq{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dq.queueFamilyIndex = qIdx; dq.queueCount = 1; dq.pQueuePriorities = &prio;
    VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dc.queueCreateInfoCount = 1; dc.pQueueCreateInfos = &dq;
    dc.pNext = &f16;
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dc, nullptr, &dev), "create device");
    VkQueue queue;
    vkGetDeviceQueue(dev, qIdx, 0, &queue);

    // Load compiled shader
    FILE* f = fopen("matmul.spv", "rb");
    if (!f) { fprintf(stderr, "[FAIL] matmul.spv not found -- run glslangValidator first\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> code(sz);
    if (fread(code.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr, "[FAIL] short read\n"); return 1; }
    fclose(f);

    VkShaderModuleCreateInfo smc{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smc.codeSize = code.size(); smc.pCode = (uint32_t*)code.data();
    VkShaderModule module;
    CHECK(vkCreateShaderModule(dev, &smc, nullptr, &module), "create shader module");

    // Buffers A, B, C (fp16 = 2 bytes each)
    VkDeviceSize bufSize = (VkDeviceSize)DIM * DIM * 2;
    VkBuffer bufs[3]; VkDeviceMemory mems[3];
    for (int i = 0; i < 3; i++) {
        VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bc.size = bufSize; bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        CHECK(vkCreateBuffer(dev, &bc, nullptr, &bufs[i]), "create buffer");
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, bufs[i], &mr);
        VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ma.allocationSize = mr.size;
        ma.memoryTypeIndex = findMemType(pd, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        CHECK(vkAllocateMemory(dev, &ma, nullptr, &mems[i]), "allocate memory");
        CHECK(vkBindBufferMemory(dev, bufs[i], mems[i], 0), "bind memory");
    }

    // Fill A and B with data (mapped, host-visible)
    uint16_t* mapA = nullptr; uint16_t* mapB = nullptr; uint16_t* mapC = nullptr;
    vkMapMemory(dev, mems[0], 0, bufSize, 0, (void**)&mapA);
    vkMapMemory(dev, mems[1], 0, bufSize, 0, (void**)&mapB);
    vkMapMemory(dev, mems[2], 0, bufSize, 0, (void**)&mapC);
    auto f32to16 = [](float v) -> uint16_t {
        // simple float->half conversion
        union { float f; uint32_t u; } conv; conv.f = v;
        uint32_t sign = (conv.u >> 16) & 0x8000u;
        int32_t exp = ((conv.u >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (conv.u >> 13) & 0x3FFu;
        if (exp <= 0) return (uint16_t)sign;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
        return (uint16_t)(sign | (exp << 10) | mant);
    };
    auto h16to32 = [](uint16_t h) -> float {
        uint32_t sign = (uint32_t)(h & 0x8000) << 16;
        int32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        if (exp == 0) return 0.0f;
        union { float f; uint32_t u; } conv;
        conv.u = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
        return conv.f;
    };
    srand(1234);
    for (uint32_t i = 0; i < DIM * DIM; i++) {
        mapA[i] = f32to16(((rand() % 200) - 100) / 100.0f);
        mapB[i] = f32to16(((rand() % 200) - 100) / 100.0f);
    }

    // Descriptor set: bindings 0,1,2 -> A,B,C
    VkDescriptorSetLayoutBinding lb[3] = {};
    for (int i = 0; i < 3; i++) { lb[i].binding = i; lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; lb[i].descriptorCount = 1; lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 3; dl.pBindings = lb;
    VkDescriptorSetLayout setLayout;
    CHECK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &setLayout), "descriptor layout");

    VkPipelineLayoutCreateInfo plc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plc.setLayoutCount = 1; plc.pSetLayouts = &setLayout;
    VkPipelineLayout pipeLayout;
    CHECK(vkCreatePipelineLayout(dev, &plc, nullptr, &pipeLayout), "pipeline layout");

    VkComputePipelineCreateInfo pc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pc.layout = pipeLayout; pc.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pc.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pc.stage.module = module; pc.stage.pName = "main";
    VkPipeline pipe;
    CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pc, nullptr, &pipe), "compute pipeline");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpc.maxSets = 1; dpc.poolSizeCount = 1; dpc.pPoolSizes = &ps;
    VkDescriptorPool pool;
    CHECK(vkCreateDescriptorPool(dev, &dpc, nullptr, &pool), "descriptor pool");
    VkDescriptorSetAllocateInfo ds{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ds.descriptorPool = pool; ds.descriptorSetCount = 1; ds.pSetLayouts = &setLayout;
    VkDescriptorSet set;
    CHECK(vkAllocateDescriptorSets(dev, &ds, &set), "descriptor set");
    VkWriteDescriptorSet w[3] = {};
    for (int i = 0; i < 3; i++) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = set; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        VkDescriptorBufferInfo bi{bufs[i], 0, bufSize};
        static VkDescriptorBufferInfo binfos[3];
        binfos[i] = bi; w[i].pBufferInfo = &binfos[i];
    }
    vkUpdateDescriptorSets(dev, 3, w, 0, nullptr);

    // Command pool + buffer: RUNS dispatches recorded back to back
    VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpc.queueFamilyIndex = qIdx;
    VkCommandPool cp;
    CHECK(vkCreateCommandPool(dev, &cpc, nullptr, &cp), "command pool");
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = cp; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    VkCommandBuffer cb;
    CHECK(vkAllocateCommandBuffers(dev, &ca, &cb), "command buffer");

    auto recordBatch = [&]() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK(vkBeginCommandBuffer(cb, &bi), "begin cmd");
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &set, 0, nullptr);
        for (uint32_t r = 0; r < RUNS; r++)
            vkCmdDispatch(cb, DIM / 8, DIM / 8, 1);
        CHECK(vkEndCommandBuffer(cb), "end cmd");
    };

    // Fence
    VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(dev, &fc, nullptr, &fence), "fence");

    // Warmup
    recordBatch();
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    CHECK(vkQueueSubmit(queue, 1, &si, fence), "submit warmup");
    CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "wait warmup");
    CHECK(vkResetFences(dev, 1, &fence), "reset fence");

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    recordBatch();
    CHECK(vkQueueSubmit(queue, 1, &si, fence), "submit batch");
    CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "wait batch");
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    // Correctness spot check: c[0] on CPU
    float expect = 0.0f;
    for (uint32_t k = 0; k < DIM; k++)
        expect += h16to32(mapA[k]) * h16to32(mapB[k * DIM]);
    float got = h16to32(mapC[0]);
    printf("[CHECK] GPU c[0]=%.2f vs CPU %.2f -> %s\n\n", got, expect,
           (fabsf(got - expect) < 2.0f) ? "MATCH (math works)" : "MISMATCH (investigate)");

    // Results
    double perMatmulSec = secs / RUNS;
    double flops = 2.0 * DIM * DIM * DIM;      // multiply-add = 2 ops per element
    double gflops = (flops / perMatmulSec) / 1e9;
    printf("=== RESULTS ===\n");
    printf("Matrix: %ux%u fp16, %u matmuls per batch\n", DIM, DIM, RUNS);
    printf("Total batch time:  %.3f ms\n", secs * 1000.0);
    printf("Time per matmul:   %.3f ms\n", perMatmulSec * 1000.0);
    printf("Throughput:        %.1f GFLOPS (fp16 scalar path)\n\n", gflops);

    printf("How to read it: a 1080p NR pass needs roughly 5000+ GFLOPS-ish of headroom\n");
    printf("to be playable. This scalar test measures a fraction of what the tensor\n");
    printf("(cooperative matrix) path can do -- it is our floor, not the ceiling.\n");

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
