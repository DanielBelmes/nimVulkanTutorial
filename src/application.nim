{.experimental: "codeReordering".}
import std/options
import glfw
import sets
import bitops
import vulkan
import vmath
from errors import RuntimeException
import types
import utils

const
    validationLayers = ["VK_LAYER_KHRONOS_validation"]
    vkInstanceExtensions = [VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME]
    deviceExtensions = [VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,VK_KHR_SWAPCHAIN_EXTENSION_NAME]
    WIDTH* = 800
    HEIGHT* = 600
    MAX_FRAMES_IN_FLIGHT: uint32 = 2

when not defined(release):
    const enableValidationLayers = true
else:
    const enableValidationLayers = false

let vertices: seq[Vertex] = @[
    vertex(vec2(0.0, -0.5),vec3(1.0, 0.0, 0.0)),
    vertex(vec2(0.5, 0.5),vec3(0.0, 1.0, 0.0)),
    vertex(vec2(-0.5, 0.5),vec3(0.0, 0.0, 1.0))
]

proc getBindingDescription(vertex: typedesc[Vertex]) : VkVertexInputBindingDescription =
    newVkVertexInputBindingDescription(
        binding = 0,
        stride = sizeof(Vertex).uint32,
        inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    )

proc getAttributeDescriptions(vertex: typedesc[Vertex]) : array[2, VkVertexInputAttributeDescription] =
    var attributeDescriptions: array[2,VkVertexInputAttributeDescription] = [
        newVkVertexInputAttributeDescription(
            binding = 0,
            location = 0,
            format = VK_FORMAT_R32G32_SFLOAT,
            offset = offsetOf(Vertex, pos).uint32
        ),
        newVkVertexInputAttributeDescription(
            binding = 0,
            location = 1,
            format = VK_FORMAT_R32G32B32_SFLOAT,
            offset = offsetOf(Vertex, color).uint32
        )
    ]
    result = attributeDescriptions

type
    VulkanTutorialApp* = ref object
        instance: VkInstance
        window: GLFWWindow
        surface: VkSurfaceKHR
        physicalDevice: VkPhysicalDevice
        deviceProperties: VkPhysicalDeviceProperties
        graphicsQueue: VkQueue
        presentQueue: VkQueue
        device: VkDevice
        swapChain: VkSwapchainKHR
        swapChainImages: seq[VkImage]
        swapChainImageFormat: VkFormat
        swapChainExtent: VkExtent2D
        swapChainImageViews: seq[VkImageView]
        pipelineLayout: VkPipelineLayout
        renderPass: VkRenderPass
        graphicsPipeline: VkPipeline
        swapChainFramebuffers: seq[VkFramebuffer]
        commandPool: VkCommandPool
        vertexBuffer: VkBuffer
        vertexBufferMemory: VkDeviceMemory
        commandBuffers: seq[VkCommandBuffer]
        imageAvailableSemaphores: seq[VkSemaphore]
        renderFinishedSemaphores: seq[VkSemaphore]
        inFlightFences: seq[VkFence]
        currentFrame: uint32
        framebufferResized: bool

proc initWindow(app: VulkanTutorialApp) =
    doAssert glfwInit()
    doAssert glfwVulkanSupported()

    glfwWindowHint(GLFWClientApi, GLFWNoApi)

    app.window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nil, nil, icon = false)
    if app.window == nil:
        quit(-1)
    setWindowUserPointer(app.window, unsafeAddr app);
    discard setFramebufferSizeCallback(app.window, cast[GLFWFramebuffersizeFun](framebufferResizeCallback))

proc framebufferResizeCallback(window: GLFWWindow, width: int, height: int) {.cdecl.} =
    let app = cast[ptr VulkanTutorialApp](getWindowUserPointer(window))
    app.framebufferResized = true

proc checkValidationLayerSupport(): bool =
    var layerCount: uint32
    discard vkEnumerateInstanceLayerProperties(addr layerCount, nil)

    var availableLayers = newSeq[VkLayerProperties](layerCount)
    discard vkEnumerateInstanceLayerProperties(addr layerCount, addr availableLayers[0])

    for layerName in validationLayers:
        var layerFound: bool = false
        for layerProperties in availableLayers:
            if cmp(layerName, cStringToString(layerProperties.layerName)) == 0:
                layerFound = true
                break

        if not layerFound:
            return false

    return true

proc createInstance(app: VulkanTutorialApp) =
    var appInfo = newVkApplicationInfo(
        pApplicationName = "NimGL Vulkan Example",
        applicationVersion = vkMakeVersion(1, 0, 0),
        pEngineName = "No Engine",
        engineVersion = vkMakeVersion(1, 0, 0),
        apiVersion = VK_API_VERSION_1_1
    )

    var glfwExtensionCount: uint32 = 0
    var glfwExtensions: cstringArray

    glfwExtensions = glfwGetRequiredInstanceExtensions(addr glfwExtensionCount)
    var extensions: seq[string]
    for ext in cstringArrayToSeq(glfwExtensions, glfwExtensionCount):
        extensions.add(ext)
    for ext in vkInstanceExtensions:
        extensions.add(ext)
    var allExtensions = allocCStringArray(extensions)


    var layerCount: uint32 = 0
    var enabledLayers: cstringArray = nil

    if enableValidationLayers:
        layerCount = uint32(validationLayers.len)
        enabledLayers = allocCStringArray(validationLayers)

    var createInfo = newVkInstanceCreateInfo(
        flags = VkInstanceCreateFlags(0x0000001),
        pApplicationInfo = addr appInfo,
        enabledExtensionCount = glfwExtensionCount + uint32(vkInstanceExtensions.len),
        ppEnabledExtensionNames = allExtensions,
        enabledLayerCount = layerCount,
        ppEnabledLayerNames = enabledLayers,
    )

    if enableValidationLayers and not checkValidationLayerSupport():
        raise newException(RuntimeException, "validation layers requested, but not available!")

    if vkCreateInstance(addr createInfo, nil, addr app.instance) != VKSuccess:
        quit("failed to create instance")

    if enableValidationLayers and not enabledLayers.isNil:
        deallocCStringArray(enabledLayers)

    if not allExtensions.isNil:
        deallocCStringArray(allExtensions)

proc createSurface(app: VulkanTutorialApp) =
    if glfwCreateWindowSurface(app.instance, app.window, nil, addr app.surface) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to create window surface")

proc checkDeviceExtensionSupport(app: VulkanTutorialApp, pDevice: VkPhysicalDevice): bool =
    var extensionCount: uint32
    discard vkEnumerateDeviceExtensionProperties(pDevice, nil, addr extensionCount, nil)
    var availableExtensions: seq[VkExtensionProperties] = newSeq[VkExtensionProperties](extensionCount)
    discard vkEnumerateDeviceExtensionProperties(pDevice, nil, addr extensionCount, addr availableExtensions[0])
    var requiredExtensions: HashSet[string] = deviceExtensions.toHashSet

    for extension in availableExtensions.mitems:
        requiredExtensions.excl(extension.extensionName.cStringToString)
    return requiredExtensions.len == 0

proc querySwapChainSupport(app: VulkanTutorialApp, pDevice: VkPhysicalDevice): SwapChainSupportDetails =
    discard vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pDevice,app.surface,addr result.capabilities)
    var formatCount: uint32
    discard vkGetPhysicalDeviceSurfaceFormatsKHR(pDevice, app.surface, addr formatCount, nil)

    if formatCount != 0:
        result.formats.setLen(formatCount)
        discard vkGetPhysicalDeviceSurfaceFormatsKHR(pDevice, app.surface, formatCount.addr, result.formats[0].addr)
    var presentModeCount: uint32
    discard vkGetPhysicalDeviceSurfacePresentModesKHR(pDevice, app.surface, presentModeCount.addr, nil)
    if presentModeCount != 0:
        result.presentModes.setLen(presentModeCount)
        discard vkGetPhysicalDeviceSurfacePresentModesKHR(pDevice, app.surface, presentModeCount.addr, result.presentModes[0].addr)

proc chooseSwapSurfaceFormat(app: VulkanTutorialApp, availableFormats: seq[VkSurfaceFormatKHR]): VkSurfaceFormatKHR =
    for format in availableFormats:
        if format.format == VK_FORMAT_B8G8R8A8_SRGB and format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
            return format
    return availableFormats[0]

proc chooseSwapPresnetMode(app: VulkanTutorialApp, availablePresentModes: seq[VkPresentModeKHR]): VkPresentModeKHR =
    for presentMode in availablePresentModes:
        if presentMode == VK_PRESENT_MODE_MAILBOX_KHR:
            return presentMode
    return VK_PRESENT_MODE_FIFO_KHR

proc chooseSwapExtent(app: VulkanTutorialApp, capabilities: VkSurfaceCapabilitiesKHR): VkExtent2D =
    if capabilities.currentExtent.width != uint32.high:
        return capabilities.currentExtent
    else:
        var width: int32
        var height: int32
        getFramebufferSize(app.window, addr width, addr height)
        result.width = clamp(cast[uint32](width),
                                capabilities.minImageExtent.width,
                                capabilities.maxImageExtent.width)
        result.height = clamp(cast[uint32](height),
                                capabilities.minImageExtent.height,
                                capabilities.maxImageExtent.height)

proc findQueueFamilies(app: VulkanTutorialApp, pDevice: VkPhysicalDevice): QueueFamilyIndices =
    var queueFamilyCount: uint32 = 0
    vkGetPhysicalDeviceQueueFamilyProperties(pDevice, addr queueFamilyCount, nil)
    var queueFamilies: seq[VkQueueFamilyProperties] = newSeq[VkQueueFamilyProperties](queueFamilyCount) # [TODO] this pattern can be templated
    vkGetPhysicalDeviceQueueFamilyProperties(pDevice, addr queueFamilyCount, addr queueFamilies[0])
    var index: uint32 = 0
    for queueFamily in queueFamilies:
        if (queueFamily.queueFlags.uint32 and VkQueueGraphicsBit.uint32) > 0'u32:
            result.graphicsFamily = some(index)
        var presentSupport: VkBool32 = VkBool32(VK_FALSE)
        discard vkGetPhysicalDeviceSurfaceSupportKHR(pDevice, index, app.surface, addr presentSupport)
        if presentSupport.ord == 1:
            result.presentFamily = some(index)

        if(result.isComplete()):
            break
        index.inc

proc isDeviceSuitable(app: VulkanTutorialApp, pDevice: VkPhysicalDevice): bool =
    var indicies: QueueFamilyIndices = app.findQueueFamilies(pDevice)
    var extensionsSupported = app.checkDeviceExtensionSupport(pDevice)
    var swapChainAdequate = false
    if extensionsSupported:
        var swapChainSupport: SwapChainSupportDetails = app.querySwapChainSupport(pDevice)
        swapChainAdequate = swapChainSupport.formats.len != 0 and swapChainSupport.presentModes.len != 0
    return indicies.isComplete and extensionsSupported and swapChainAdequate

proc pickPhysicalDevice(app: VulkanTutorialApp) =
    var deviceCount: uint32 = 0
    discard vkEnumeratePhysicalDevices(app.instance, addr deviceCount, nil)
    if(deviceCount == 0):
        raise newException(RuntimeException, "failed to find GPUs with Vulkan support!")
    var pDevices: seq[VkPhysicalDevice] = newSeq[VkPhysicalDevice](deviceCount)
    discard vkEnumeratePhysicalDevices(app.instance, addr deviceCount, addr pDevices[0])
    for pDevice in pDevices:
        if app.isDeviceSuitable(pDevice):
            app.physicalDevice = pDevice
            vkGetPhysicalDeviceProperties(app.physicalDevice, app.deviceProperties.addr)
            return

    raise newException(RuntimeException, "failed to find a suitable GPU!")

proc createLogicalDevice(app: VulkanTutorialApp) =
    let
        indices = app.findQueueFamilies(app.physicalDevice)
        uniqueQueueFamilies = [indices.graphicsFamily.get, indices.presentFamily.get].toHashSet
    var
        queuePriority = 1f
        queueCreateInfos = newSeq[VkDeviceQueueCreateInfo]()

    for queueFamily in uniqueQueueFamilies:
        let deviceQueueCreateInfo: VkDeviceQueueCreateInfo = newVkDeviceQueueCreateInfo(
            queueFamilyIndex = queueFamily,
            queueCount = 1,
            pQueuePriorities = queuePriority.addr
        )
        queueCreateInfos.add(deviceQueueCreateInfo)

    var
        deviceFeatures = newSeq[VkPhysicalDeviceFeatures](1)
        deviceExts = allocCStringArray(deviceExtensions)
        deviceCreateInfo = newVkDeviceCreateInfo(
            pQueueCreateInfos = queueCreateInfos[0].addr,
            queueCreateInfoCount = queueCreateInfos.len.uint32,
            pEnabledFeatures = deviceFeatures[0].addr,
            enabledExtensionCount = deviceExtensions.len.uint32,
            enabledLayerCount = 0,
            ppEnabledLayerNames = nil,
            ppEnabledExtensionNames = deviceExts
        )

    if vkCreateDevice(app.physicalDevice, deviceCreateInfo.addr, nil, app.device.addr) != VKSuccess:
        echo "failed to create logical device"

    vkGetDeviceQueue(app.device, indices.graphicsFamily.get, 0, addr app.graphicsQueue)
    vkGetDeviceQueue(app.device, indices.presentFamily.get, 0, addr app.presentQueue)

    if not deviceExts.isNil:
        deallocCStringArray(deviceExts)


proc createSwapChain(app: VulkanTutorialApp) =
    let swapChainSupport: SwapChainSupportDetails = app.querySwapChainSupport(app.physicalDevice)

    let surfaceFormat: VkSurfaceFormatKHR = app.chooseSwapSurfaceFormat(swapChainSupport.formats)
    let presentMode: VkPresentModeKHR = app.chooseSwapPresnetMode(swapChainSupport.presentModes)
    let extent: VkExtent2D = app.chooseSwapExtent(swapChainSupport.capabilities)

    var imageCount: uint32 = swapChainSupport.capabilities.minImageCount + 1 # request one extra per recommended settings

    if swapChainSupport.capabilities.maxImageCount > 0 and imageCount > swapChainSupport.capabilities.maxImageCount:
        imageCount = swapChainSupport.capabilities.maxImageCount

    var createInfo = VkSwapchainCreateInfoKHR(
        sType: VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        surface: app.surface,
        minImageCount: imageCount,
        imageFormat: surfaceFormat.format,
        imageColorSpace: surfaceFormat.colorSpace,
        imageExtent: extent,
        imageArrayLayers: 1,
        imageUsage: VkImageUsageFlags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
        preTransform: swapChainSupport.capabilities.currentTransform,
        compositeAlpha: VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        presentMode: presentMode,
        clipped: VKBool32(VK_TRUE),
        oldSwapchain: VkSwapchainKHR(VK_NULL_HANDLE)
    )
    let indices = app.findQueueFamilies(app.physicalDevice)
    var queueFamilyIndicies = [indices.graphicsFamily.get, indices.presentFamily.get]

    if indices.graphicsFamily.get != indices.presentFamily.get:
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT
        createInfo.queueFamilyIndexCount = 2
        createInfo.pQueueFamilyIndices = queueFamilyIndicies[0].addr
    else:
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE
        createInfo.queueFamilyIndexCount = 0
        createInfo.pQueueFamilyIndices = nil

    if vkCreateSwapchainKHR(app.device, addr createInfo, nil, addr app.swapChain) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to create swap chain!")
    discard vkGetSwapchainImagesKHR(app.device, app.swapChain, addr imageCount, nil)
    app.swapChainImages.setLen(imageCount)
    discard vkGetSwapchainImagesKHR(app.device, app.swapChain, addr imageCount, addr app.swapChainImages[0])
    app.swapChainImageFormat = surfaceFormat.format
    app.swapChainExtent = extent

proc createImageViews(app: VulkanTutorialApp) =
    app.swapChainImageViews.setLen(app.swapChainImages.len)
    for index, swapChainImage in app.swapChainImages:
        var createInfo = newVkImageViewCreateInfo(
            image = swapChainImage,
            viewType = VK_IMAGE_VIEW_TYPE_2D,
            format = app.swapChainImageFormat,
            components = newVkComponentMapping(VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY),
            subresourceRange = newVkImageSubresourceRange(aspectMask = VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT), 0.uint32, 1.uint32, 0.uint32, 1.uint32)
        )
        if vkCreateImageView(app.device, addr createInfo, nil, addr app.swapChainImageViews[index]) != VK_SUCCESS:
            raise newException(RuntimeException, "failed to create image views")

proc createShaderModule(app: VulkanTutorialApp, code: string) : VkShaderModule =
    var createInfo = newVkShaderModuleCreateInfo(
        codeSize = code.len.uint32,
        pCode = cast[ptr uint32](code[0].unsafeAddr) #Hopefully reading bytecode as string is alright
    )
    if vkCreateShaderModule(app.device, addr createInfo, nil, addr result) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to create shader module")

proc createRenderPass(app: VulkanTutorialApp) =
    var
        colorAttachment: VkAttachmentDescription = newVkAttachmentDescription(
            format = app.swapChainImageFormat,
            samples = VK_SAMPLE_COUNT_1_BIT,
            loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )
        colorAttachmentRef: VkAttachmentReference = newVkAttachmentReference(
            attachment = 0,
            layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        subpass = VkSubpassDescription(
            pipelineBindPoint: VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount: 1,
            pColorAttachments: addr colorAttachmentRef,
        )
        dependency: VkSubpassDependency = VkSubpassDependency(
            srcSubpass: VK_SUBPASS_EXTERNAL,
            dstSubpass: 0,
            srcStageMask: VkPipelineStageFlags(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
            srcAccessMask: VkAccessFlags(0),
            dstStageMask: VkPipelineStageFlags(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
            dstAccessMask: VkAccessFlags(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT),
        )
        renderPassInfo: VkRenderPassCreateInfo = newVkRenderPassCreateInfo(
            attachmentCount = 1,
            pAttachments = addr colorAttachment,
            subpassCount = 1,
            pSubpasses = addr subpass,
            dependencyCount = 1,
            pDependencies = addr dependency,
        )
    if vkCreateRenderPass(app.device, addr renderPassInfo, nil, addr app.renderPass) != VK_SUCCESS:
        quit("failed to create render pass")

proc createGraphicsPipeline(app: VulkanTutorialApp) =
    const
        vertShaderCode: string = staticRead("./shaders/vert.spv")
        fragShaderCode: string = staticRead("./shaders/frag.spv")
    var
        vertShaderModule: VkShaderModule = app.createShaderModule(vertShaderCode)
        fragShaderModule: VkShaderModule = app.createShaderModule(fragShaderCode)
        vertShaderStageInfo: VkPipelineShaderStageCreateInfo = newVkPipelineShaderStageCreateInfo(
            stage = VK_SHADER_STAGE_VERTEX_BIT,
            module = vertShaderModule,
            pName = "main",
            pSpecializationInfo = nil
        )
        fragShaderStageInfo: VkPipelineShaderStageCreateInfo = newVkPipelineShaderStageCreateInfo(
            stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            module = fragShaderModule,
            pName = "main",
            pSpecializationInfo = nil
        )
        shaderStages: array[2, VkPipelineShaderStageCreateInfo] = [vertShaderStageInfo, fragShaderStageInfo]
        dynamicStates: array[2, VkDynamicState] = [VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR]
        dynamicState: VkPipelineDynamicStateCreateInfo = newVkPipelineDynamicStateCreateInfo(
            dynamicStateCount = dynamicStates.len.uint32,
            pDynamicStates = addr dynamicStates[0]
        )
        bindingDescription = Vertex.getBindingDescription()
        attributeDescriptions = Vertex.getAttributeDescriptions()
        vertexInputInfo: VkPipelineVertexInputStateCreateInfo = newVkPipelineVertexInputStateCreateInfo(
            vertexBindingDescriptionCount = 1,
            pVertexBindingDescriptions = addr bindingDescription,
            vertexAttributeDescriptionCount = cast[uint32](attributeDescriptions.len),
            pVertexAttributeDescriptions = addr attributeDescriptions[0]
        )
        inputAssembly: VkPipelineInputAssemblyStateCreateInfo = newVkPipelineInputAssemblyStateCreateInfo(
            topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable = VkBool32(VK_FALSE)
        )
        viewport: VkViewPort = newVkViewport(
            x = 0.float,
            y = 0.float,
            width = app.swapChainExtent.width.float32,
            height = app.swapChainExtent.height.float32,
            minDepth = 0.float,
            maxDepth = 1.float
        )
        scissor: VkRect2D = newVkRect2D(
            offset = newVkOffset2D(0,0),
            extent = app.swapChainExtent
        )
        viewportState: VkPipelineViewportStateCreateInfo = newVkPipelineViewportStateCreateInfo(
            viewportCount = 1,
            pViewports = addr viewport,
            scissorCount = 1,
            pScissors = addr scissor
        )
        rasterizer: VkPipelineRasterizationStateCreateInfo = newVkPipelineRasterizationStateCreateInfo(
            depthClampEnable = VkBool32(VK_FALSE),
            rasterizerDiscardEnable = VkBool32(VK_FALSE),
            polygonMode = VK_POLYGON_MODE_FILL,
            lineWidth = 1.float,
            cullMode = VkCullModeFlags(VK_CULL_MODE_BACK_BIT),
            frontface = VK_FRONT_FACE_CLOCKWISE,
            depthBiasEnable = VKBool32(VK_FALSE),
            depthBiasConstantFactor = 0.float,
            depthBiasClamp = 0.float,
            depthBiasSlopeFactor = 0.float,
        )
        multisampling: VkPipelineMultisampleStateCreateInfo = newVkPipelineMultisampleStateCreateInfo(
            sampleShadingEnable = VkBool32(VK_FALSE),
            rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            minSampleShading = 1.float,
            pSampleMask = nil,
            alphaToCoverageEnable = VkBool32(VK_FALSE),
            alphaToOneEnable = VkBool32(VK_FALSE)
        )
        # [NOTE] Not doing VkPipelineDepthStencilStateCreateInfo because we don't have a depth or stencil buffer yet
        colorBlendAttachment: VkPipelineColorBlendAttachmentState = newVkPipelineColorBlendAttachmentState(
            colorWriteMask = VkColorComponentFlags(bitor(VK_COLOR_COMPONENT_R_BIT.int32, bitor(VK_COLOR_COMPONENT_G_BIT.int32, bitor(VK_COLOR_COMPONENT_B_BIT.int32, VK_COLOR_COMPONENT_A_BIT.int32)))),
            blendEnable = VkBool32(VK_FALSE),
            srcColorBlendFactor = VK_BLEND_FACTOR_ONE, # optional
            dstColorBlendFactor = VK_BLEND_FACTOR_ZERO, # optional
            colorBlendOp = VK_BLEND_OP_ADD, # optional
            srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, # optional
            dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, # optional
            alphaBlendOp = VK_BLEND_OP_ADD, # optional
        )
        colorBlending: VkPipelineColorBlendStateCreateInfo = newVkPipelineColorBlendStateCreateInfo(
            logicOpEnable = VkBool32(VK_FALSE),
            logicOp = VK_LOGIC_OP_COPY, # optional
            attachmentCount = 1,
            pAttachments = colorBlendAttachment.addr,
            blendConstants = [0f, 0f, 0f, 0f], # optional
        )
        pipelineLayoutInfo: VkPipelineLayoutCreateInfo = newVkPipelineLayoutCreateInfo(
            setLayoutCount = 0, # optional
            pSetLayouts = nil, # optional
            pushConstantRangeCount = 0, # optional
            pPushConstantRanges = nil, # optional
        )
    if vkCreatePipelineLayout(app.device, pipelineLayoutInfo.addr, nil, addr app.pipelineLayout) != VK_SUCCESS:
        quit("failed to create pipeline layout")
    var
        pipelineInfo: VkGraphicsPipelineCreateInfo = newVkGraphicsPipelineCreateInfo(
            stageCount = shaderStages.len.uint32,
            pStages = shaderStages[0].addr,
            pVertexInputState = vertexInputInfo.addr,
            pInputAssemblyState = inputAssembly.addr,
            pViewportState = viewportState.addr,
            pRasterizationState = rasterizer.addr,
            pMultisampleState = multisampling.addr,
            pDepthStencilState = nil, # optional
            pColorBlendState = colorBlending.addr,
            pDynamicState = dynamicState.addr, # optional
            pTessellationState = nil,
            layout = app.pipelineLayout,
            renderPass = app.renderPass,
            subpass = 0,
            basePipelineHandle = VkPipeline(0), # optional
            basePipelineIndex = -1, # optional
        )
    if vkCreateGraphicsPipelines(app.device, VkPipelineCache(0), 1, pipelineInfo.addr, nil, addr app.graphicsPipeline) != VK_SUCCESS:
        quit("fialed to create graphics pipeline")
    vkDestroyShaderModule(app.device, vertShaderModule, nil)
    vkDestroyShaderModule(app.device, fragShaderModule, nil)

proc createFrameBuffers(app: VulkanTutorialApp) =
    app.swapChainFramebuffers.setLen(app.swapChainImageViews.len)

    for index, view in app.swapChainImageViews:
        var
            attachments = [app.swapChainImageViews[index]]
            framebufferInfo = newVkFramebufferCreateInfo(
                sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass = app.renderPass,
                attachmentCount = attachments.len.uint32,
                pAttachments = attachments[0].addr,
                width = app.swapChainExtent.width,
                height = app.swapChainExtent.height,
                layers = 1,
            )
        if vkCreateFramebuffer(app.device, framebufferInfo.addr, nil, addr app.swapChainFramebuffers[index]) != VK_SUCCESS:
            quit("failed to create framebuffer")

proc cleanupSwapChain(app: VulkanTutorialApp) =
    for framebuffer in app.swapChainFramebuffers:
        vkDestroyFramebuffer(app.device, framebuffer, nil)
    for imageView in app.swapChainImageViews:
        vkDestroyImageView(app.device, imageView, nil)
    vkDestroySwapchainKHR(app.device, app.swapChain, nil)

proc recreateSwapChain(app: VulkanTutorialApp) =
    var
        width: int32 = 0
        height: int32 = 0
    getFramebufferSize(app.window, addr width, addr height)
    while width == 0 or height == 0:
        getFramebufferSize(app.window, addr width, addr height)
        glfwWaitEvents()
    discard vkDeviceWaitIdle(app.device)

    app.cleanupSwapChain()

    app.createSwapChain()
    app.createImageViews()
    app.createFramebuffers()

proc createCommandPool(app: VulkanTutorialApp) =
    var
        indicies: QueueFamilyIndices = app.findQueueFamilies(app.physicalDevice) # I should just save this info. Does it change?
        poolInfo: VkCommandPoolCreateInfo = newVkCommandPoolCreateInfo(
            flags = VkCommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT),
            queueFamilyIndex = indicies.graphicsFamily.get
        )
    if vkCreateCommandPool(app.device, addr poolInfo, nil, addr app.commandPool) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to create command pool!")

proc findMemoryType(app: VulkanTutorialApp, typeFilter: uint32, properties: VkMemoryPropertyFlags) : uint32 =
    var memProperties: VkPhysicalDeviceMemoryProperties
    vkGetPhysicalDeviceMemoryProperties(app.physicalDevice, addr memProperties);
    for i in 0..<memProperties.memoryTypeCount:
        if (typeFilter and (1 shl i).uint32).bool and (memProperties.memoryTypes[i].propertyFlags.uint32 and properties.uint32) == properties.uint32:
            return i
    raise newException(RuntimeException, "failed to find suitable memory type!")


proc createVertexBuffer(app: VulkanTutorialApp) =
    let
        bufferInfo: VkBufferCreateInfo = VkBufferCreateInfo(
            sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size: (sizeof(vertices[0]) * vertices.len).VkDeviceSize,
            usage: VK_BUFFER_USAGE_VERTEX_BUFFER_BIT.VkBufferUsageFlags,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE
        )

    if vkCreateBuffer(app.device, addr bufferInfo, nil, addr app.vertexBuffer) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to create vertex buffer!")
    var memRequirements: VkMemoryRequirements
    vkGetBufferMemoryRequirements(app.device, app.vertexBuffer, addr memRequirements)

    let allocInfo: VkMemoryAllocateInfo = newVkMemoryAllocateInfo(
        allocationSize = memRequirements.size,
        memoryTypeIndex = app.findMemoryType(memRequirements.memoryTypeBits.uint32, (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT).VkMemoryPropertyFlags)
    )

    if vkAllocateMemory(app.device, addr allocInfo, nil, addr app.vertexBufferMemory) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to allocate vertex buffer memory!");

    if vkBindBufferMemory(app.device, app.vertexBuffer, app.vertexBufferMemory, 0.VkDeviceSize) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to bind buffer memory")

    var data : pointer
    if vkMapMemory(app.device, app.vertexBufferMemory, 0.VkDeviceSize, bufferInfo.size, 0.VkMemoryMapFlags, addr data) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to map memory")
    copyMem(data, addr vertices[0], bufferInfo.size.uint32)
    let
        alignedSize : uint32 = (bufferInfo.size.uint32 - 1) - ((bufferInfo.size.uint32 - 1) mod app.deviceProperties.limits.nonCoherentAtomSize.uint32) + app.deviceProperties.limits.nonCoherentAtomSize.uint32
        vertexRange: VkMappedMemoryRange = newVkMappedMemoryRange(
            memory = app.vertexBufferMemory,
            offset = 0.VkDeviceSize,
            size   = alignedSize.VkDeviceSize
        )
    discard vkFlushMappedMemoryRanges(app.device, 1, addr vertexRange)
    vkUnmapMemory(app.device, app.vertexBufferMemory)

proc createCommandBuffers(app: VulkanTutorialApp) =
    app.commandBuffers.setLen(MAX_FRAMES_IN_FLIGHT)
    let allocInfo: VkCommandBufferAllocateInfo = newVkCommandBufferAllocateInfo(
        commandPool = app.commandPool,
        level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount = cast[uint32](app.commandBuffers.len)
    )
    if vkAllocateCommandBuffers(app.device, addr allocInfo, addr app.commandBuffers[0]) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to allocate command buffers!")

proc recordCommandBuffer(app: VulkanTutorialApp, commandBuffer: VkCommandBuffer, imageIndex: uint32) =
    let beginInfo: VkCommandBufferBeginInfo = newVkCommandBufferBeginInfo(
        flags = VkCommandBufferUsageFlags(0),
        pInheritanceInfo = nil
    )
    if vkBeginCommandBuffer(commandBuffer, addr beginInfo) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to begin recording command buffer!")

    let
        clearColor: VkClearValue = VkClearValue(color: VkClearColorValue(float32: [0f, 0f, 0f, 1f]))
        renderPassInfo: VkRenderPassBeginInfo = newVkRenderPassBeginInfo(
            renderPass = app.renderPass,
            framebuffer = app.swapChainFrameBuffers[imageIndex],
            renderArea = VkRect2D(
                offset: VkOffset2d(x: 0,y: 0),
                extent: app.swapChainExtent
            ),
            clearValueCount = 1,
            pClearValues = addr clearColor
        )
    vkCmdBeginRenderPass(commandBuffer, renderPassInfo.addr, VK_SUBPASS_CONTENTS_INLINE)
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app.graphicsPipeline)
    let
        viewport: VkViewport = newVkViewport(
            x = 0f,
            y = 0f,
            width = app.swapChainExtent.width.float32,
            height = app.swapChainExtent.height.float32,
            minDepth = 0f,
            maxDepth = 1f
        )
        scissor: VkRect2D = newVkRect2D(
            offset = VkOffset2D(x: 0, y: 0),
            extent = app.swapChainExtent
        )
        vertexBuffers: array[1, VkBuffer] = [app.vertexBuffer]
        offsets: array[1, VkDeviceSize] = [0.VkDeviceSize]
    vkCmdSetViewport(commandBuffer, 0, 1, addr viewport)
    vkCmdSetScissor(commandBuffer, 0, 1, addr scissor)
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, addr vertexBuffers[0], addr offsets[0])
    vkCmdDraw(commandBuffer, vertices.len.uint32, 1, 0, 0)
    vkCmdEndRenderPass(commandBuffer)
    if vkEndCommandBuffer(commandBuffer) != VK_SUCCESS:
        quit("failed to record command buffer")

proc createSyncObjects(app: VulkanTutorialApp) =
    app.imageAvailableSemaphores.setLen(MAX_FRAMES_IN_FLIGHT)
    app.renderFinishedSemaphores.setLen(MAX_FRAMES_IN_FLIGHT)
    app.inFlightFences.setLen(MAX_FRAMES_IN_FLIGHT)
    let
        semaphoreInfo: VkSemaphoreCreateInfo = newVkSemaphoreCreateInfo()
        fenceInfo: VkFenceCreateInfo = newVkFenceCreateInfo(
            flags = VkFenceCreateFlags(VK_FENCE_CREATE_SIGNALED_BIT)
        )
    for i in countup(0,cast[int](MAX_FRAMES_IN_FLIGHT-1)):
        if  (vkCreateSemaphore(app.device, addr semaphoreInfo, nil, addr app.imageAvailableSemaphores[i]) != VK_SUCCESS) or 
            (vkCreateSemaphore(app.device, addr semaphoreInfo, nil, addr app.renderFinishedSemaphores[i]) != VK_SUCCESS) or 
            (vkCreateFence(app.device, addr fenceInfo, nil, addr app.inFlightFences[i]) != VK_SUCCESS):
                raise newException(RuntimeException, "failed to create sync Objects!")

proc drawFrame(app: VulkanTutorialApp) =
    discard vkWaitForFences(app.device, 1, addr app.inFlightFences[app.currentFrame], VkBool32(VK_TRUE), uint64.high)
    var imageIndex: uint32
    let imageResult: VkResult = vkAcquireNextImageKHR(app.device, app.swapChain, uint64.high, app.imageAvailableSemaphores[app.currentFrame], VkFence(0), addr imageIndex)
    if imageResult == VK_ERROR_OUT_OF_DATE_KHR:
        app.recreateSwapChain();
        return
    elif (imageResult != VK_SUCCESS and imageResult != VK_SUBOPTIMAL_KHR):
        raise newException(RuntimeException, "failed to acquire swap chain image!")

    # Only reset the fence if we are submitting work
    discard vkResetFences(app.device, 1 , addr app.inFlightFences[app.currentFrame])

    discard vkResetCommandBuffer(app.commandBuffers[app.currentFrame], VkCommandBufferResetFlags(0))
    app.recordCommandBuffer(app.commandBuffers[app.currentFrame], imageIndex)
    let
        waitSemaphores: array[1, VkSemaphore] = [app.imageAvailableSemaphores[app.currentFrame]]
        waitStages: array[1, VkPipelineStageFlags] = [VkPipelineStageFlags(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)]
        signalSemaphores: array[1, VkSemaphore] = [app.renderFinishedSemaphores[app.currentFrame]]
        submitInfo: VkSubmitInfo = newVkSubmitInfo(
            waitSemaphoreCount = waitSemaphores.len.uint32,
            pWaitSemaphores = addr waitSemaphores[0],
            pWaitDstStageMask = addr waitStages[0],
            commandBufferCount = 1,
            pCommandBuffers = addr app.commandBuffers[app.currentFrame],
            signalSemaphoreCount = 1,
            pSignalSemaphores = addr signalSemaphores[0]
        )
    if vkQueueSubmit(app.graphicsQueue, 1, addr submitInfo, app.inFlightFences[app.currentFrame]) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to submit draw command buffer")
    let
        swapChains: array[1, VkSwapchainKHR] = [app.swapChain]
        presentInfo: VkPresentInfoKHR = newVkPresentInfoKHR(
            waitSemaphoreCount = 1,
            pWaitSemaphores = addr signalSemaphores[0],
            swapchainCount = 1,
            pSwapchains = addr swapChains[0],
            pImageIndices = addr imageIndex,
            pResults = nil
        )
    let queueResult = vkQueuePresentKHR(app.presentQueue, addr presentInfo)
    if queueResult == VK_ERROR_OUT_OF_DATE_KHR or queueResult == VK_SUBOPTIMAL_KHR or app.framebufferResized:
        app.framebufferResized = false
        app.recreateSwapChain();
    elif queueResult != VK_SUCCESS:
        raise newException(RuntimeException, "failed to present swap chain image!")
    app.currentFrame = (app.currentFrame + 1).mod(MAX_FRAMES_IN_FLIGHT)

proc initVulkan(app: VulkanTutorialApp) =
    app.createInstance()
    app.createSurface()
    app.pickPhysicalDevice()
    app.createLogicalDevice()
    app.createSwapChain()
    app.createImageViews()
    app.createRenderPass()
    app.createGraphicsPipeline()
    app.createFrameBuffers()
    app.createCommandPool()
    app.createVertexBuffer()
    app.createCommandBuffers()
    app.createSyncObjects()
    app.framebufferResized = false
    app.currentFrame = 0

proc mainLoop(app: VulkanTutorialApp) =
    while not windowShouldClose(app.window):
        glfwPollEvents()
        app.drawFrame()
    discard vkDeviceWaitIdle(app.device);

proc cleanup(app: VulkanTutorialApp) =
    for i in countup(0,cast[int](MAX_FRAMES_IN_FLIGHT-1)):
        vkDestroySemaphore(app.device, app.imageAvailableSemaphores[i], nil)
        vkDestroySemaphore(app.device, app.renderFinishedSemaphores[i], nil)
        vkDestroyFence(app.device, app.inFlightFences[i], nil)
    vkDestroyCommandPool(app.device, app.commandPool, nil)
    vkDestroyPipeline(app.device, app.graphicsPipeline, nil)
    vkDestroyPipelineLayout(app.device, app.pipelineLayout, nil)
    vkDestroyRenderPass(app.device, app.renderPass, nil)
    app.cleanupSwapChain()
    vkDestroyBuffer(app.device, app.vertexBuffer, nil);
    vkFreeMemory(app.device, app.vertexBufferMemory, nil);
    vkDestroyDevice(app.device, nil) #destroy device before instance
    vkDestroySurfaceKHR(app.instance, app.surface, nil)
    vkDestroyInstance(app.instance, nil)
    app.window.destroyWindow()
    glfwTerminate()

proc run*(app: VulkanTutorialApp) =
    app.initWindow()
    app.initVulkan()
    app.mainLoop()
    app.cleanup()