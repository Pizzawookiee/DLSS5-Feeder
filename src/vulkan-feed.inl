// Experimental Vulkan backend for DLSS5-Feeder.
//
// Architecture:
//   ReShade Vulkan frame + LaunchPad-produced DLSS5_MV / DLSS5_Depth
//      -> Vulkan blit/copy into D3D12-shared images
//      -> private D3D12 NGX DLAA evaluation
//      -> shared output
//      -> Vulkan fullscreen blit back into the ReShade target.
//
// This deliberately follows the proven Vulkan<->D3D12 raster-host pattern in
// chasmlol/fnv-dlss5. It keeps the RenoDX D3D12 interception contract while
// using LaunchPad guides consistently with the existing feeder backends.

struct VulkanFeedState
{
    struct ImportedImage
    {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    } images[SLOT_COUNT];

    bool active = false;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    reshade::api::format backbuffer_format = reshade::api::format::unknown;

    HMODULE vulkan_module = nullptr;
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr = nullptr;
    PFN_vkCreateImage CreateImage = nullptr;
    PFN_vkDestroyImage DestroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements = nullptr;
    PFN_vkGetMemoryWin32HandlePropertiesKHR GetMemoryWin32HandlePropertiesKHR = nullptr;
    PFN_vkAllocateMemory AllocateMemory = nullptr;
    PFN_vkFreeMemory FreeMemory = nullptr;
    PFN_vkBindImageMemory BindImageMemory = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyImage CmdCopyImage = nullptr;
    PFN_vkCmdBlitImage CmdBlitImage = nullptr;
    PFN_vkCreateSemaphore CreateSemaphore = nullptr;
    PFN_vkDestroySemaphore DestroySemaphore = nullptr;
    PFN_vkImportSemaphoreWin32HandleKHR ImportSemaphoreWin32HandleKHR = nullptr;
    PFN_vkQueueSubmit QueueSubmit = nullptr;
    PFN_vkQueueWaitIdle QueueWaitIdle = nullptr;

    ID3D12Fence *interop_fence = nullptr;
    HANDLE interop_handle = nullptr;
    VkSemaphore interop_semaphore = VK_NULL_HANDLE;
    uint64_t interop_value = 0;
};

static VulkanFeedState gv;

static bool LoadVulkanFunctions(VkDevice device)
{
    gv.vulkan_module = GetModuleHandleW(L"vulkan-1.dll");
    if (gv.vulkan_module == nullptr)
        gv.vulkan_module = LoadLibraryW(L"vulkan-1.dll");
    if (gv.vulkan_module == nullptr)
    {
        Log("[feed-vk] vulkan-1.dll unavailable");
        return false;
    }

    gv.GetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        GetProcAddress(gv.vulkan_module, "vkGetDeviceProcAddr"));
    if (gv.GetDeviceProcAddr == nullptr)
        return false;

#define FEED_VK_LOAD(name) \
    gv.name = reinterpret_cast<PFN_vk##name>(gv.GetDeviceProcAddr(device, "vk" #name)); \
    if (gv.name == nullptr) { Log("[feed-vk] missing vk" #name " (required extension may not be enabled)"); return false; }

    FEED_VK_LOAD(CreateImage);
    FEED_VK_LOAD(DestroyImage);
    FEED_VK_LOAD(GetImageMemoryRequirements);
    FEED_VK_LOAD(GetMemoryWin32HandlePropertiesKHR);
    FEED_VK_LOAD(AllocateMemory);
    FEED_VK_LOAD(FreeMemory);
    FEED_VK_LOAD(BindImageMemory);
    FEED_VK_LOAD(CmdPipelineBarrier);
    FEED_VK_LOAD(CmdCopyImage);
    FEED_VK_LOAD(CmdBlitImage);
    FEED_VK_LOAD(CreateSemaphore);
    FEED_VK_LOAD(DestroySemaphore);
    FEED_VK_LOAD(ImportSemaphoreWin32HandleKHR);
    FEED_VK_LOAD(QueueSubmit);
    FEED_VK_LOAD(QueueWaitIdle);
#undef FEED_VK_LOAD

    return true;
}

static uint32_t FirstMemoryType(uint32_t bits)
{
    for (uint32_t i = 0; i != 32; ++i)
        if ((bits & (1u << i)) != 0)
            return i;
    return UINT32_MAX;
}

static void ImportedBarrier(
    VkCommandBuffer cmd,
    VulkanFeedState::ImportedImage &image,
    VkImageLayout new_layout,
    VkPipelineStageFlags src_stage,
    VkPipelineStageFlags dst_stage,
    VkAccessFlags src_access,
    VkAccessFlags dst_access)
{
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = image.layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    gv.CmdPipelineBarrier(
        cmd, src_stage, dst_stage, 0,
        0, nullptr, 0, nullptr, 1, &barrier);
    image.layout = new_layout;
}

static void ReleaseVulkanImports()
{
    if (gv.device == VK_NULL_HANDLE)
        return;

    for (int i = 0; i < SLOT_COUNT; ++i)
    {
        if (gv.images[i].image != VK_NULL_HANDLE)
            gv.DestroyImage(gv.device, gv.images[i].image, nullptr);
        if (gv.images[i].memory != VK_NULL_HANDLE)
            gv.FreeMemory(gv.device, gv.images[i].memory, nullptr);
        gv.images[i] = {};
    }
}

static void ReleaseVulkanFrameResources()
{
    if (gv.active && gv.queue != VK_NULL_HANDLE && gv.QueueWaitIdle != nullptr)
        gv.QueueWaitIdle(gv.queue);

    // FNV precedent: destroy the Vulkan import before releasing the D3D12 allocation.
    ReleaseVulkanImports();
    ReleaseFrameResources();
    gv.backbuffer_format = reshade::api::format::unknown;
}

static void ShutdownVulkanSession()
{
    if (!gv.active && gv.device == VK_NULL_HANDLE)
        return;

    Log("[feed-vk] shutting down Vulkan interop session");

    if (gv.queue != VK_NULL_HANDLE && gv.QueueWaitIdle != nullptr)
        gv.QueueWaitIdle(gv.queue);
    DrainGpu();
    ReleaseVulkanImports();

    if (gv.device != VK_NULL_HANDLE &&
        gv.interop_semaphore != VK_NULL_HANDLE &&
        gv.DestroySemaphore != nullptr)
        gv.DestroySemaphore(gv.device, gv.interop_semaphore, nullptr);
    gv.interop_semaphore = VK_NULL_HANDLE;

    SafeRelease(gv.interop_fence);
    if (gv.interop_handle != nullptr)
    {
        CloseHandle(gv.interop_handle);
        gv.interop_handle = nullptr;
    }

    gv.active = false;
    gv.device = VK_NULL_HANDLE;
    gv.queue = VK_NULL_HANDLE;
    gv.interop_value = 0;
    gv.backbuffer_format = reshade::api::format::unknown;

    // Releases feature, D3D12 shared resources and the private D3D12 NGX host.
    ShutdownSession();
}

static bool ImportSharedD3D12Image(
    int slot, UINT width, UINT height, VkFormat format)
{
    HANDLE handle = nullptr;
    if (!DuplicateHandle(
            GetCurrentProcess(), g.shared[slot],
            GetCurrentProcess(), &handle,
            0, FALSE, DUPLICATE_SAME_ACCESS))
    {
        Log("[feed-vk] DuplicateHandle(%s) failed (%lu)",
            kSlotName[slot], GetLastError());
        return false;
    }

    VkExternalMemoryImageCreateInfo external = {};
    external.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;

    VkImageCreateInfo image_info = {};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.pNext = &external;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format;
    image_info.extent = { width, height, 1 };
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    VkResult vr = gv.CreateImage(gv.device, &image_info, nullptr, &image);
    if (vr != VK_SUCCESS)
    {
        CloseHandle(handle);
        Log("[feed-vk] vkCreateImage(%s) failed (%d)",
            kSlotName[slot], static_cast<int>(vr));
        return false;
    }

    VkMemoryRequirements requirements = {};
    gv.GetImageMemoryRequirements(gv.device, image, &requirements);

    VkMemoryWin32HandlePropertiesKHR handle_props = {};
    handle_props.sType = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR;
    vr = gv.GetMemoryWin32HandlePropertiesKHR(
        gv.device,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
        handle,
        &handle_props);
    if (vr != VK_SUCCESS)
    {
        gv.DestroyImage(gv.device, image, nullptr);
        CloseHandle(handle);
        Log("[feed-vk] vkGetMemoryWin32HandlePropertiesKHR(%s) failed (%d)",
            kSlotName[slot], static_cast<int>(vr));
        return false;
    }

    const uint32_t type_index =
        FirstMemoryType(requirements.memoryTypeBits & handle_props.memoryTypeBits);
    if (type_index == UINT32_MAX)
    {
        gv.DestroyImage(gv.device, image, nullptr);
        CloseHandle(handle);
        Log("[feed-vk] no compatible imported memory type for %s", kSlotName[slot]);
        return false;
    }

    VkMemoryDedicatedAllocateInfo dedicated = {};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.image = image;

    VkImportMemoryWin32HandleInfoKHR import_info = {};
    import_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    import_info.pNext = &dedicated;
    import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    import_info.handle = handle;

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = &import_info;
    alloc_info.allocationSize = requirements.size;
    alloc_info.memoryTypeIndex = type_index;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    vr = gv.AllocateMemory(gv.device, &alloc_info, nullptr, &memory);
    CloseHandle(handle);
    if (vr != VK_SUCCESS)
    {
        gv.DestroyImage(gv.device, image, nullptr);
        Log("[feed-vk] vkAllocateMemory(import %s) failed (%d)",
            kSlotName[slot], static_cast<int>(vr));
        return false;
    }

    vr = gv.BindImageMemory(gv.device, image, memory, 0);
    if (vr != VK_SUCCESS)
    {
        gv.FreeMemory(gv.device, memory, nullptr);
        gv.DestroyImage(gv.device, image, nullptr);
        Log("[feed-vk] vkBindImageMemory(%s) failed (%d)",
            kSlotName[slot], static_cast<int>(vr));
        return false;
    }

    gv.images[slot].image = image;
    gv.images[slot].memory = memory;
    gv.images[slot].layout = VK_IMAGE_LAYOUT_UNDEFINED;

    Log("[feed-vk] imported %-6s %ux%u via D3D12_RESOURCE",
        kSlotName[slot], width, height);
    return true;
}

static bool MakeVulkanSharedSlot(
    int slot,
    UINT width,
    UINT height,
    DXGI_FORMAT dxgi_format,
    VkFormat vk_format,
    bool uav)
{
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = dxgi_format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS |
                 (uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                      : D3D12_RESOURCE_FLAG_NONE);

    HRESULT hr = g.dev12->CreateCommittedResource(
        &heap,
        D3D12_HEAP_FLAG_SHARED,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        __uuidof(ID3D12Resource),
        reinterpret_cast<void **>(&g.tex12[slot]));
    if (SUCCEEDED(hr))
        hr = g.dev12->CreateSharedHandle(
            g.tex12[slot], nullptr, GENERIC_ALL, nullptr, &g.shared[slot]);

    if (FAILED(hr))
    {
        Log("[feed-vk] creating shared D3D12 %s failed 0x%08X",
            kSlotName[slot], hr);
        return false;
    }

    return ImportSharedD3D12Image(slot, width, height, vk_format);
}

static bool CreateVulkanInteropFence()
{
    HRESULT hr = g.dev12->CreateFence(
        0,
        D3D12_FENCE_FLAG_SHARED,
        __uuidof(ID3D12Fence),
        reinterpret_cast<void **>(&gv.interop_fence));
    if (SUCCEEDED(hr))
        hr = g.dev12->CreateSharedHandle(
            gv.interop_fence, nullptr, GENERIC_ALL, nullptr, &gv.interop_handle);

    if (FAILED(hr) || gv.interop_handle == nullptr)
    {
        Log("[feed-vk] shared D3D12 fence creation failed 0x%08X", hr);
        return false;
    }

    VkSemaphoreTypeCreateInfo timeline = {};
    timeline.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline.initialValue = 0;

    VkSemaphoreCreateInfo semaphore_info = {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_info.pNext = &timeline;

    VkResult vr = gv.CreateSemaphore(
        gv.device, &semaphore_info, nullptr, &gv.interop_semaphore);
    if (vr != VK_SUCCESS)
    {
        Log("[feed-vk] timeline semaphore creation failed (%d)",
            static_cast<int>(vr));
        return false;
    }

    HANDLE handle = nullptr;
    if (!DuplicateHandle(
            GetCurrentProcess(), gv.interop_handle,
            GetCurrentProcess(), &handle,
            0, FALSE, DUPLICATE_SAME_ACCESS))
        return false;

    VkImportSemaphoreWin32HandleInfoKHR import_info = {};
    import_info.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
    import_info.semaphore = gv.interop_semaphore;
    import_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
    import_info.handle = handle;

    vr = gv.ImportSemaphoreWin32HandleKHR(gv.device, &import_info);
    CloseHandle(handle);

    if (vr != VK_SUCCESS)
    {
        Log("[feed-vk] importing D3D12 fence into Vulkan failed (%d)",
            static_cast<int>(vr));
        return false;
    }

    Log("[feed-vk] cross-API D3D12-fence/timeline-semaphore ready");
    return true;
}

static bool SubmitVulkanSignal(uint64_t value)
{
    VkTimelineSemaphoreSubmitInfo timeline = {};
    timeline.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline.signalSemaphoreValueCount = 1;
    timeline.pSignalSemaphoreValues = &value;

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = &timeline;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &gv.interop_semaphore;

    const VkResult vr = gv.QueueSubmit(gv.queue, 1, &submit, VK_NULL_HANDLE);
    if (vr != VK_SUCCESS)
        Log("[feed-vk] signal submit failed (%d)", static_cast<int>(vr));
    return vr == VK_SUCCESS;
}

static bool SubmitVulkanWait(uint64_t value)
{
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

    VkTimelineSemaphoreSubmitInfo timeline = {};
    timeline.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline.waitSemaphoreValueCount = 1;
    timeline.pWaitSemaphoreValues = &value;

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = &timeline;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &gv.interop_semaphore;
    submit.pWaitDstStageMask = &stage;

    const VkResult vr = gv.QueueSubmit(gv.queue, 1, &submit, VK_NULL_HANDLE);
    if (vr != VK_SUCCESS)
        Log("[feed-vk] wait submit failed (%d)", static_cast<int>(vr));
    return vr == VK_SUCCESS;
}

static bool InitSessionVK(reshade::api::effect_runtime *rt)
{
    using namespace reshade::api;

    device *dev_api = rt->get_device();
    command_queue *queue_api = rt->get_command_queue();
    if (queue_api == nullptr)
        return false;

    const VkDevice vk_device =
        reinterpret_cast<VkDevice>(dev_api->get_native());
    const VkQueue vk_queue =
        reinterpret_cast<VkQueue>(queue_api->get_native());

    if (gv.active && gv.device == vk_device && g.session_ready)
        return true;

    if (gv.active)
        ShutdownVulkanSession();
    else if (g.session_ready)
        ShutdownSession();

    gv.device = vk_device;
    gv.queue = vk_queue;
    g.rs_queue = queue_api;

    if (vk_device == VK_NULL_HANDLE ||
        vk_queue == VK_NULL_HANDLE ||
        !LoadVulkanFunctions(vk_device))
    {
        FeedDisable("required Vulkan external-memory/semaphore functions are unavailable");
        return false;
    }

    LUID luid = {};
    if (!dev_api->get_property(device_properties::adapter_luid, &luid))
    {
        FeedDisable("Vulkan adapter LUID is unavailable");
        return false;
    }

    HMODULE d3d12 = LoadLibraryW(L"d3d12.dll");
    HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
    auto create_device = d3d12
        ? reinterpret_cast<PFN_D3D12CreateDevice_>(
              GetProcAddress(d3d12, "D3D12CreateDevice"))
        : nullptr;
    using CreateFactoryFn = HRESULT (WINAPI *)(REFIID, void **);
    auto create_factory = dxgi
        ? reinterpret_cast<CreateFactoryFn>(
              GetProcAddress(dxgi, "CreateDXGIFactory1"))
        : nullptr;

    if (create_device == nullptr || create_factory == nullptr)
    {
        FeedDisable("D3D12/DXGI interop entry points are unavailable");
        return false;
    }

    IDXGIFactory4 *factory = nullptr;
    HRESULT hr = create_factory(
        __uuidof(IDXGIFactory4),
        reinterpret_cast<void **>(&factory));
    if (FAILED(hr) || factory == nullptr)
        return false;

    IDXGIAdapter1 *matched = nullptr;
    for (UINT index = 0; ; ++index)
    {
        IDXGIAdapter1 *candidate = nullptr;
        if (factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND)
            break;

        DXGI_ADAPTER_DESC1 desc = {};
        if (SUCCEEDED(candidate->GetDesc1(&desc)) &&
            desc.AdapterLuid.HighPart == luid.HighPart &&
            desc.AdapterLuid.LowPart == luid.LowPart)
        {
            matched = candidate;
            break;
        }
        candidate->Release();
    }
    factory->Release();

    if (matched == nullptr)
    {
        FeedDisable("no D3D12 adapter matched the Vulkan adapter");
        return false;
    }

    DXGI_ADAPTER_DESC1 adapter_desc = {};
    matched->GetDesc1(&adapter_desc);
    Log("[feed-vk] matched adapter: %ls", adapter_desc.Description);

    hr = create_device(
        matched,
        D3D_FEATURE_LEVEL_11_0,
        __uuidof(ID3D12Device),
        reinterpret_cast<void **>(&g.dev12));
    matched->Release();

    if (FAILED(hr) || g.dev12 == nullptr)
    {
        FeedDisable("matching D3D12 device creation failed");
        return false;
    }

    g.dev12_owned = true;
    g.dev11 = nullptr;

    wchar_t data_path[MAX_PATH] = {};
    GetModuleFileNameW(g_self, data_path, MAX_PATH);
    if (wchar_t *slash = wcsrchr(data_path, L'\\'))
        *(slash + 1) = L'\0';

    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Init(
        0x1000000ULL,
        data_path,
        g.dev12,
        nullptr,
        NVSDK_NGX_Version_API);
    Log("[feed-vk] NVSDK_NGX_D3D12_Init -> 0x%08X (%s)",
        result, NgxResultName(result));

    if (NVSDK_NGX_FAILED(result))
        result = NVSDK_NGX_D3D12_Init_with_ProjectID(
            "a0f57b54-1daf-4934-90ae-c4035c19df04",
            NVSDK_NGX_ENGINE_TYPE_CUSTOM,
            "1.0",
            data_path,
            g.dev12,
            nullptr,
            NVSDK_NGX_Version_API);

    if (NVSDK_NGX_FAILED(result))
    {
        FeedDisable("NGX failed on the Vulkan-matched D3D12 device");
        return false;
    }
    g.ngx_inited = true;

    result = NVSDK_NGX_D3D12_AllocateParameters(&g.params);
    if (NVSDK_NGX_FAILED(result) || g.params == nullptr)
    {
        FeedDisable("NGX parameter allocation failed");
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = g.dev12->CreateCommandQueue(
        &queue_desc,
        __uuidof(ID3D12CommandQueue),
        reinterpret_cast<void **>(&g.queue));
    if (FAILED(hr) || g.queue == nullptr)
        return false;

    for (int i = 0; i < Feed::kFrames; ++i)
    {
        hr = g.dev12->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            __uuidof(ID3D12CommandAllocator),
            reinterpret_cast<void **>(&g.alloc[i]));
        if (FAILED(hr))
            return false;
    }

    hr = g.dev12->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        g.alloc[0],
        nullptr,
        __uuidof(ID3D12GraphicsCommandList),
        reinterpret_cast<void **>(&g.list));
    if (FAILED(hr) || g.list == nullptr)
        return false;
    g.list->Close();

    hr = g.dev12->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        __uuidof(ID3D12Fence),
        reinterpret_cast<void **>(&g.fence12));
    g.fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (FAILED(hr) || g.fence12 == nullptr || g.fence_event == nullptr)
        return false;

    if (!CreateVulkanInteropFence())
    {
        FeedDisable("Vulkan/D3D12 shared-fence setup failed");
        return false;
    }

    g.session_ready = true;
    gv.active = true;
    g.frame_ready = false;
    g.need_reset = true;
    g.create_grace = 0;

    Log("[feed-vk] interop session ready: VkDevice=%p VkQueue=%p D3D12=%p",
        reinterpret_cast<void *>(gv.device),
        reinterpret_cast<void *>(gv.queue),
        reinterpret_cast<void *>(g.dev12));
    return true;
}

static bool BuildResourcesVK(
    UINT width,
    UINT height,
    reshade::api::format backbuffer_format)
{
    const DXGI_FORMAT marker = static_cast<DXGI_FORMAT>(
        static_cast<uint32_t>(backbuffer_format));

    if (g.session_ready &&
        g_cfg.mode >= 2 &&
        g.feature != nullptr &&
        gv.images[SLOT_COLOR].image != VK_NULL_HANDLE &&
        width == g.width &&
        height == g.height &&
        marker == g.bb_fmt)
        return RecreateFeatureOnly(width, height);

    ReleaseVulkanFrameResources();

    g.width = width;
    g.height = height;
    g.bb_fmt = marker;
    gv.backbuffer_format = backbuffer_format;

    // FNV raster-host precedent: fixed FP16 color/output interop avoids
    // BGRA/UAV restrictions; Vulkan blits perform edge format conversion.
    g.color_fmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
    g.output_fmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
    g.hdr = g_cfg.hdr >= 0
        ? g_cfg.hdr != 0
        : (backbuffer_format == reshade::api::format::r16g16b16a16_float ||
           backbuffer_format == reshade::api::format::r11g11b10_float);

    const bool inverted = g_cfg.depth_inverted >= 0
        ? g_cfg.depth_inverted != 0
        : g.depth_reversed;

    const bool ok =
        MakeVulkanSharedSlot(
            SLOT_COLOR, width, height,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            VK_FORMAT_R16G16B16A16_SFLOAT, false) &&
        MakeVulkanSharedSlot(
            SLOT_OUTPUT, width, height,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            VK_FORMAT_R16G16B16A16_SFLOAT, true) &&
        MakeVulkanSharedSlot(
            SLOT_DEPTH, width, height,
            DXGI_FORMAT_R32_FLOAT,
            VK_FORMAT_R32_SFLOAT, false) &&
        MakeVulkanSharedSlot(
            SLOT_MV, width, height,
            DXGI_FORMAT_R16G16_FLOAT,
            VK_FORMAT_R16G16_SFLOAT, false);

    if (!ok)
    {
        ReleaseVulkanFrameResources();
        return false;
    }

    if (g_cfg.mode < 2)
    {
        g.frame_ready = true;
        g.need_reset = true;
        Log("[feed-vk] transport ready (mode %d, no NGX feature)", g_cfg.mode);
        return true;
    }

    bool crashed = false;
    if (!CreateDlssFeature(width, height, inverted, &crashed))
    {
        if (crashed)
            FeedDisable("creating the Vulkan-hosted DLSS feature crashed");
        return false;
    }

    return true;
}

static void CopyVkImage(
    VkCommandBuffer cmd,
    VkImage source,
    VkImageLayout source_layout,
    VkImage dest,
    VkImageLayout dest_layout,
    UINT width,
    UINT height)
{
    VkImageCopy region = {};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.extent = { width, height, 1 };

    gv.CmdCopyImage(
        cmd, source, source_layout, dest, dest_layout, 1, &region);
}

static void BlitVkImage(
    VkCommandBuffer cmd,
    VkImage source,
    VkImageLayout source_layout,
    VkImage dest,
    VkImageLayout dest_layout,
    UINT width,
    UINT height)
{
    VkImageBlit region = {};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;
    region.srcOffsets[1] = {
        static_cast<int32_t>(width),
        static_cast<int32_t>(height),
        1
    };
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.dstOffsets[1] = {
        static_cast<int32_t>(width),
        static_cast<int32_t>(height),
        1
    };

    gv.CmdBlitImage(
        cmd,
        source, source_layout,
        dest, dest_layout,
        1, &region,
        VK_FILTER_NEAREST);
}

static bool EvaluateVulkanInterop(bool reset)
{
    const uint64_t ready_for_d3d12 = ++gv.interop_value;
    const uint64_t ready_for_vulkan = ++gv.interop_value;

    if (!SubmitVulkanSignal(ready_for_d3d12))
        return false;

    if (FAILED(g.queue->Wait(gv.interop_fence, ready_for_d3d12)))
        return false;

    if (!BeginCommands())
        return false;

    Barrier(g.tex12[SLOT_COLOR], D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(g.tex12[SLOT_DEPTH], D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(g.tex12[SLOT_MV], D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(g.tex12[SLOT_OUTPUT], D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    NVSDK_NGX_D3D12_DLSS_Eval_Params eval = {};
    eval.Feature.pInColor = g.tex12[SLOT_COLOR];
    eval.Feature.pInOutput = g.tex12[SLOT_OUTPUT];
    eval.Feature.InSharpness = 0.0f;
    eval.pInDepth = g.tex12[SLOT_DEPTH];
    eval.pInMotionVectors = g.tex12[SLOT_MV];
    eval.InJitterOffsetX = 0.0f;
    eval.InJitterOffsetY = 0.0f;
    eval.InRenderSubrectDimensions.Width = g.width;
    eval.InRenderSubrectDimensions.Height = g.height;
    eval.InReset = reset ? 1 : 0;
    eval.InMVScaleX = g_cfg.mv_scale_x;
    eval.InMVScaleY = g_cfg.mv_scale_y;
    eval.InPreExposure = 1.0f;
    eval.InExposureScale = 1.0f;

    DWORD exception_code = 0;
    const NVSDK_NGX_Result result =
        SafeEvaluateDLSS(&eval, &exception_code);

    if (exception_code == 0)
    {
        Barrier(g.tex12[SLOT_COLOR],
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON);
        Barrier(g.tex12[SLOT_DEPTH],
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON);
        Barrier(g.tex12[SLOT_MV],
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON);
        Barrier(g.tex12[SLOT_OUTPUT],
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON);
        EndCommands();
    }
    else
    {
        AbortCommands();
        Log("[feed-vk] evaluate raised exception 0x%08X", exception_code);
    }

    // Never leave the Vulkan queue permanently waiting if NGX fails.
    g.queue->Signal(gv.interop_fence, ready_for_vulkan);
    if (!SubmitVulkanWait(ready_for_vulkan))
        return false;

    if (exception_code != 0)
    {
        FeedDisable("the Vulkan-hosted DLSS evaluate crashed");
        g.frame_ready = false;
        return false;
    }

    if (NVSDK_NGX_FAILED(result))
    {
        Log("[feed-vk] evaluate failed 0x%08X (%s)",
            result, NgxResultName(result));
        FeedFail("Vulkan-hosted evaluate");
        g.frame_ready = false;
        return false;
    }

    return true;
}

static void FeedFrameVK(
    reshade::api::effect_runtime *rt,
    reshade::api::command_list *cl,
    reshade::api::resource_view rtv)
{
    using namespace reshade::api;

    if ((g.frames_done % 60) == 0 && CfgReload())
        g.frame_ready = false;
    if (!g_cfg.enabled || g_cfg.mode == 0)
        return;

    device *dev_api = rt->get_device();

    resource_view mv_srv = {}, mv_srgb = {};
    resource_view depth_srv = {}, depth_srgb = {};
    if (g.mv_var.handle != 0)
        rt->get_texture_binding(g.mv_var, &mv_srv, &mv_srgb);
    if (g.depth_var.handle != 0)
        rt->get_texture_binding(g.depth_var, &depth_srv, &depth_srgb);

    if (mv_srv.handle == 0 || depth_srv.handle == 0)
    {
        if (!g.missing_reported)
        {
            g.missing_reported = true;
            Warn("Vulkan: DLSS5_Feed guide textures are missing. "
                 "Enable MartysMods LaunchPad above DLSS5_Feed.");
        }
        return;
    }

    const resource backbuffer = dev_api->get_resource_from_view(rtv);
    const resource motion = dev_api->get_resource_from_view(mv_srv);
    const resource depth = dev_api->get_resource_from_view(depth_srv);
    if (backbuffer.handle == 0 || motion.handle == 0 || depth.handle == 0)
        return;

    const resource_desc color_desc = dev_api->get_resource_desc(backbuffer);
    const resource_desc mv_desc = dev_api->get_resource_desc(motion);
    const resource_desc depth_desc = dev_api->get_resource_desc(depth);

    const UINT width = color_desc.texture.width;
    const UINT height = color_desc.texture.height;

    if (width == 0 || height == 0 ||
        mv_desc.texture.width != width ||
        mv_desc.texture.height != height ||
        depth_desc.texture.width != width ||
        depth_desc.texture.height != height ||
        color_desc.texture.samples != 1 ||
        mv_desc.texture.format != format::r16g16_float ||
        depth_desc.texture.format != format::r32_float)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            Log("[feed-vk] input mismatch: color %ux%u samples=%u, "
                "mv %ux%u fmt=%u, depth %ux%u fmt=%u",
                width, height, color_desc.texture.samples,
                mv_desc.texture.width, mv_desc.texture.height,
                static_cast<unsigned>(mv_desc.texture.format),
                depth_desc.texture.width, depth_desc.texture.height,
                static_cast<unsigned>(depth_desc.texture.format));
        }
        return;
    }

    if (!InitSessionVK(rt))
        return;

    const bool needs_build =
        !g.frame_ready ||
        width != g.width ||
        height != g.height ||
        color_desc.texture.format != gv.backbuffer_format;

    if (needs_build && g.create_grace < g_cfg.create_delay)
    {
        if (++g.create_grace == 1)
            Log("[feed-vk] holding feature build for %d frames",
                g_cfg.create_delay);
        return;
    }

    if (needs_build &&
        !BuildResourcesVK(width, height, color_desc.texture.format))
    {
        FeedFail("Vulkan interop resource build");
        return;
    }

    VkCommandBuffer command_buffer =
        reinterpret_cast<VkCommandBuffer>(cl->get_native());
    if (command_buffer == VK_NULL_HANDLE)
        return;

    const VkImage color_image =
        reinterpret_cast<VkImage>(static_cast<uintptr_t>(backbuffer.handle));
    const VkImage mv_image =
        reinterpret_cast<VkImage>(static_cast<uintptr_t>(motion.handle));
    const VkImage depth_image =
        reinterpret_cast<VkImage>(static_cast<uintptr_t>(depth.handle));

    // Source state tracking stays in the ReShade API.
    {
        const resource resources[3] = { backbuffer, motion, depth };
        const resource_usage before[3] = {
            resource_usage::render_target,
            resource_usage::shader_resource,
            resource_usage::shader_resource
        };
        const resource_usage after[3] = {
            resource_usage::copy_source,
            resource_usage::copy_source,
            resource_usage::copy_source
        };
        cl->barrier(3, resources, before, after);
    }

    ImportedBarrier(
        command_buffer, gv.images[SLOT_COLOR],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        gv.images[SLOT_COLOR].layout == VK_IMAGE_LAYOUT_UNDEFINED
            ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, VK_ACCESS_TRANSFER_WRITE_BIT);

    ImportedBarrier(
        command_buffer, gv.images[SLOT_MV],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        gv.images[SLOT_MV].layout == VK_IMAGE_LAYOUT_UNDEFINED
            ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, VK_ACCESS_TRANSFER_WRITE_BIT);

    ImportedBarrier(
        command_buffer, gv.images[SLOT_DEPTH],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        gv.images[SLOT_DEPTH].layout == VK_IMAGE_LAYOUT_UNDEFINED
            ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, VK_ACCESS_TRANSFER_WRITE_BIT);

    // FNV precedent: color bridge is FP16. A Vulkan blit handles format conversion.
    BlitVkImage(
        command_buffer,
        color_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        gv.images[SLOT_COLOR].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        width, height);

    // LaunchPad guides exactly match the bridge formats.
    CopyVkImage(
        command_buffer,
        mv_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        gv.images[SLOT_MV].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        width, height);
    CopyVkImage(
        command_buffer,
        depth_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        gv.images[SLOT_DEPTH].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        width, height);

    ImportedBarrier(
        command_buffer, gv.images[SLOT_COLOR],
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    ImportedBarrier(
        command_buffer, gv.images[SLOT_MV],
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    ImportedBarrier(
        command_buffer, gv.images[SLOT_DEPTH],
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);

    if (gv.images[SLOT_OUTPUT].layout == VK_IMAGE_LAYOUT_UNDEFINED)
        ImportedBarrier(
            command_buffer, gv.images[SLOT_OUTPUT],
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);

    {
        const resource resources[3] = { backbuffer, motion, depth };
        const resource_usage before[3] = {
            resource_usage::copy_source,
            resource_usage::copy_source,
            resource_usage::copy_source
        };
        const resource_usage after[3] = {
            resource_usage::render_target,
            resource_usage::shader_resource,
            resource_usage::shader_resource
        };
        cl->barrier(3, resources, before, after);
    }

    // Submit LaunchPad/feed output + input transfers before D3D12 can consume them.
    g.rs_queue->flush_immediate_command_list();

    const bool reset = g.need_reset || g_cfg.reset_every;
    g.need_reset = false;

    if (g_cfg.mode >= 2 && !EvaluateVulkanInterop(reset))
        return;

    // ReShade now exposes its fresh immediate command buffer. The raw Vulkan
    // timeline wait was submitted before it, so queue order makes D3D12 output visible.
    command_buffer = reinterpret_cast<VkCommandBuffer>(cl->get_native());
    if (command_buffer == VK_NULL_HANDLE)
        return;

    {
        const resource resources[1] = { backbuffer };
        const resource_usage before[1] = { resource_usage::render_target };
        const resource_usage after[1] = { resource_usage::copy_dest };
        cl->barrier(1, resources, before, after);
    }

    VulkanFeedState::ImportedImage &composite =
        g_cfg.mode >= 2 ? gv.images[SLOT_OUTPUT] : gv.images[SLOT_COLOR];

    ImportedBarrier(
        command_buffer, composite,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT);

    // Fullscreen Vulkan composite: blit instead of raw resource replacement.
    BlitVkImage(
        command_buffer,
        composite.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        color_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        width, height);

    ImportedBarrier(
        command_buffer, composite,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);

    {
        const resource resources[1] = { backbuffer };
        const resource_usage before[1] = { resource_usage::copy_dest };
        const resource_usage after[1] = { resource_usage::render_target };
        cl->barrier(1, resources, before, after);
    }

    const UINT64 frame = ++g.frames_done;
    g.consecutive_fails = 0;
    if (frame <= static_cast<UINT64>(g_cfg.log_frames) ||
        (frame % 1800) == 0)
        Log("[feed-vk] frame %llu delivered (%ux%u reset=%d)",
            frame, width, height, reset ? 1 : 0);
}
