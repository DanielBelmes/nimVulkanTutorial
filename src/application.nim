{.experimental: "codeReordering".}
import std/options
import sets
import bitops
import vulkan
import vmath
from errors import RuntimeException
import types
import utils
import std/monotimes
import std/times
import stb_nim/stb_image
import std/sequtils
import std/tables
import objLoader
from nglfw as glfw import nil

when defined macosx:
    const
        vkInstanceExtensions = [VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME]
        deviceExtensions = [VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,VK_KHR_SWAPCHAIN_EXTENSION_NAME]
else:
    const
        vkInstanceExtensions :array[0,string]= []
        deviceExtensions = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]

const
    validationLayers = ["VK_LAYER_KHRONOS_validation"]
    WIDTH* = 800
    HEIGHT* = 600
    MAX_FRAMES_IN_FLIGHT: uint32 = 2
    MODEL_PATH = "models/viking_room.obj"
    TEXTURE_PATH = "textures/viking_room.png"

when not defined(release):
    const enableValidationLayers = true
else:
    const enableValidationLayers = false

let
    startTime: MonoTime = getMonoTime()

proc getBindingDescription(vertex: typedesc[Vertex]) : VkVertexInputBindingDescription =
    newVkVertexInputBindingDescription(
        binding = 0,
        stride = sizeof(Vertex).uint32,
        inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    )

proc getAttributeDescriptions(vertex: typedesc[Vertex]) : array[3, VkVertexInputAttributeDescription] =
    var attributeDescriptions: array[3,VkVertexInputAttributeDescription] = [
        newVkVertexInputAttributeDescription(
            binding = 0,
            location = 0,
            format = VK_FORMAT_R32G32B32_SFLOAT,
            offset = offsetOf(Vertex, pos).uint32
        ),
        newVkVertexInputAttributeDescription(
            binding = 0,
            location = 1,
            format = VK_FORMAT_R32G32B32_SFLOAT,
            offset = offsetOf(Vertex, color).uint32
        ),
        newVkVertexInputAttributeDescription(
            binding = 0,
            location = 2,
            format = VK_FORMAT_R32G32_SFLOAT,
            offset = offsetOf(Vertex, texCoord).uint32
        )
    ]
    result = attributeDescriptions

type
    VulkanTutorialApp* = ref object
        instance: VkInstance
        window: glfw.Window
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
        descriptorSetLayout: VkDescriptorSetLayout
        pipelineLayout: VkPipelineLayout
        renderPass: VkRenderPass
        graphicsPipeline: VkPipeline
        swapChainFramebuffers: seq[VkFramebuffer]
        commandPool: VkCommandPool
        vertexBuffer: VkBuffer
        vertexBufferMemory: VkDeviceMemory
        indexBuffer: VkBuffer # [TODO] combine vertex and indexBuffer and use offset
        indexBufferMemory: VkDeviceMemory
        uniformBuffers: seq[VkBuffer]
        uniformBuffersMemory: seq[VkDeviceMemory]
        uniformBuffersMapped: seq[pointer]
        textureImage: VkImage
        textureImageMemory: VkDeviceMemory
        textureImageView: VkImageView
        textureSampler: VkSampler
        depthImage: VkImage
        depthImageMemory: VkDeviceMemory
        depthImageView: VkImageView
        descriptorPool: VkDescriptorPool
        descriptorSets: seq[VkDescriptorSet]
        commandBuffers: seq[VkCommandBuffer]
        imageAvailableSemaphores: seq[VkSemaphore]
        renderFinishedSemaphores: seq[VkSemaphore]
        inFlightFences: seq[VkFence]
        currentFrame: uint32
        framebufferResized: bool
        vertices: seq[Vertex] = @[]
        sceneIndices: seq[uint32] = @[]

proc initWindow(app: var VulkanTutorialApp) =
    doAssert glfw.init()
    doAssert glfw.vulkanSupported()

    glfw.windowHint(glfw.ClientApi, glfw.NoApi)

    app.window = glfw.createWindow(WIDTH.cint, HEIGHT.cint, "Vulkan", nil, nil)
    doAssert app.window != nil
    glfw.setWindowUserPointer(app.window, addr app);
    discard glfw.setKeyCallback(app.window, keyCallback)
    discard glfw.setFramebufferSizeCallback(app.window, framebufferResizeCallback)

proc framebufferResizeCallback(window: glfw.Window, width: int32, height: int32) {.cdecl.} =
    let app = cast[ptr VulkanTutorialApp](glfw.getWindowUserPointer(window))
    if app.isNil: return
    app.framebufferResized = true

proc keyCallback (window :glfw.Window; key, code, action, mods :int32) :void {.cdecl.}=
  ## GLFW Keyboard Input Callback
  if (key == glfw.KeyEscape and action == glfw.Press):
    glfw.setWindowShouldClose(window, true)

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

    glfwExtensions = glfw.getRequiredInstanceExtensions(addr glfwExtensionCount)
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
        pApplicationInfo = addr appInfo,
        enabledExtensionCount = glfwExtensionCount + uint32(vkInstanceExtensions.len),
        ppEnabledExtensionNames = allExtensions,
        enabledLayerCount = layerCount,
        ppEnabledLayerNames = enabledLayers,
    )

    when defined macosx:
        createInfo.flags = VkInstanceCreateFlags(0x0000001)

    if enableValidationLayers and not checkValidationLayerSupport():
        raise newException(RuntimeException, "validation layers requested, but not available!")

    if vkCreateInstance(addr createInfo, nil, addr app.instance) != VKSuccess:
        quit("failed to create instance")

    if enableValidationLayers and not enabledLayers.isNil:
        deallocCStringArray(enabledLayers)

    if not allExtensions.isNil:
        deallocCStringArray(allExtensions)

proc createSurface(app: VulkanTutorialApp) =
    if glfw.createWindowSurface(app.instance, app.window, nil, addr app.surface) != VK_SUCCESS:
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
        glfw.getFramebufferSize(app.window, addr width, addr height)
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
    var
        indices: QueueFamilyIndices = app.findQueueFamilies(pDevice)
        extensionsSupported = app.checkDeviceExtensionSupport(pDevice)
        swapChainAdequate = false
        supportedFeatures: VkPhysicalDeviceFeatures
    vkGetPhysicalDeviceFeatures(pDevice, addr supportedFeatures)
    if extensionsSupported:
        var swapChainSupport: SwapChainSupportDetails = app.querySwapChainSupport(pDevice)
        swapChainAdequate = swapChainSupport.formats.len != 0 and swapChainSupport.presentModes.len != 0
    return indices.isComplete and extensionsSupported and swapChainAdequate and supportedFeatures.samplerAnisotropy.bool

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
        deviceFeatures: VkPhysicalDeviceFeatures = VkPhysicalDeviceFeatures(samplerAnisotropy: VK_TRUE.VkBool32)
        deviceExts = allocCStringArray(deviceExtensions)
        deviceCreateInfo = newVkDeviceCreateInfo(
            pQueueCreateInfos = queueCreateInfos[0].addr,
            queueCreateInfoCount = queueCreateInfos.len.uint32,
            pEnabledFeatures = addr deviceFeatures,
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
    var queueFamilyindices = [indices.graphicsFamily.get, indices.presentFamily.get]

    if indices.graphicsFamily.get != indices.presentFamily.get:
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT
        createInfo.queueFamilyIndexCount = 2
        createInfo.pQueueFamilyIndices = queueFamilyindices[0].addr
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
        app.swapChainImageViews[index] = app.createImageView(app.swapChainImages[index], app.swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT.VkImageAspectFlags)

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
        depthAttachment: VkAttachmentDescription = newVkAttachmentDescription(
            format = app.findDepthFormat(),
            samples = VK_SAMPLE_COUNT_1_BIT,
            loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )
        depthAttachmentRef: VkAttachmentReference = VkAttachmentReference(
            attachment: 1,
            layout: VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )
        subpass = VkSubpassDescription(
            pipelineBindPoint: VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount: 1,
            pColorAttachments: addr colorAttachmentRef,
            pDepthStencilAttachment: addr depthAttachmentRef
        )
        attachments: array[2, VkAttachmentDescription] = [colorAttachment,depthAttachment]
        dependency: VkSubpassDependency = VkSubpassDependency(
            srcSubpass: VK_SUBPASS_EXTERNAL,
            dstSubpass: 0,
            srcStageMask: VkPipelineStageFlags(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT or VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT),
            srcAccessMask: VkAccessFlags(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
            dstStageMask: VkPipelineStageFlags(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT or VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT),
            dstAccessMask: VkAccessFlags(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT or VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
        )
        renderPassInfo: VkRenderPassCreateInfo = newVkRenderPassCreateInfo(
            attachmentCount = attachments.len.uint32,
            pAttachments = addr attachments[0],
            subpassCount = 1,
            pSubpasses = addr subpass,
            dependencyCount = 1,
            pDependencies = addr dependency,
        )
    if vkCreateRenderPass(app.device, addr renderPassInfo, nil, addr app.renderPass) != VK_SUCCESS:
        quit("failed to create render pass")

proc createDescriptorSetLayout(app: VulkanTutorialApp) =
    let
        uboLayoutBinding: VkDescriptorSetLayoutBinding = newVkDescriptorSetLayoutBinding(
            binding = 0, # same as in vert shader
            descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount = 1,
            stageFlags = VK_SHADER_STAGE_VERTEX_BIT.VkShaderStageFlags,
            pImmutableSamplers = nil
        )
        samplerLayoutBinding: VkDescriptorSetLayoutBinding = newVkDescriptorSetLayoutBinding(
            binding = 1, # same as in vert shader
            descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount = 1,
            stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT.VkShaderStageFlags,
            pImmutableSamplers = nil
        )
        bindings : array[2,VkDescriptorSetLayoutBinding] = [uboLayoutBinding,samplerLayoutBinding]
    let
        layoutInfo: VkDescriptorSetLayoutCreateInfo = newVkDescriptorSetLayoutCreateInfo(
            bindingCount = bindings.len.uint32,
            pBindings = addr bindings[0]
        )

    if vkCreateDescriptorSetLayout(app.device, addr layoutInfo, nil, addr app.descriptorSetLayout) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to create descriptor set layout")

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
            frontface = VK_FRONT_FACE_COUNTER_CLOCKWISE,
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
        depthStencil: VkPipelineDepthStencilStateCreateInfo = newVkPipelineDepthStencilStateCreateInfo(
            depthTestEnable = VkBool32(VK_TRUE),
            depthWriteEnable = VkBool32(VK_TRUE),
            depthCompareOp = VK_COMPARE_OP_LESS,
            depthBoundsTestEnable = VkBool32(VK_FALSE),
            minDepthBounds = 0.0f, # optional
            maxDepthBounds = 1.0f, # optional
            stencilTestEnable = VkBool32(VK_FALSE),
            front = VkStencilOpState(),
            back = VkStencilOpState()
        )
        colorBlendAttachment: VkPipelineColorBlendAttachmentState = newVkPipelineColorBlendAttachmentState(
            colorWriteMask = VkColorComponentFlags(bitor(VK_COLOR_COMPONENT_R_BIT.int32, bitor(VK_COLOR_COMPONENT_G_BIT.int32, bitor(VK_COLOR_COMPONENT_B_BIT.int32, VK_COLOR_COMPONENT_A_BIT.int32)))),
            blendEnable = VkBool32(VK_FALSE), # remember to enable to use transparent images
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
            setLayoutCount = 1,
            pSetLayouts = addr app.descriptorSetLayout,
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
            pDepthStencilState = addr depthStencil,
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
            attachments : array[2,VkImageView] = [app.swapChainImageViews[index],app.depthImageView]
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
    glfw.getFramebufferSize(app.window, addr width, addr height)
    while width == 0 or height == 0:
        glfw.getFramebufferSize(app.window, addr width, addr height)
        glfw.waitEvents()
    discard vkDeviceWaitIdle(app.device)

    app.cleanupSwapChain()

    app.createSwapChain()
    app.createImageViews()
    app.createDepthResources()
    app.createFramebuffers()

proc createCommandPool(app: VulkanTutorialApp) =
    var
        indices: QueueFamilyIndices = app.findQueueFamilies(app.physicalDevice) # I should just save this info. Does it change?
        poolInfo: VkCommandPoolCreateInfo = newVkCommandPoolCreateInfo(
            flags = VkCommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT),
            queueFamilyIndex = indices.graphicsFamily.get
        )
    if vkCreateCommandPool(app.device, addr poolInfo, nil, addr app.commandPool) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to create command pool!")

proc hasStencilComponent(format: VkFormat): bool =
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT or format == VK_FORMAT_D24_UNORM_S8_UINT

proc findSupportedFormat(app: VulkanTutorialApp, canidates: seq[VkFormat], tiling: VkImageTiling, features: VkFormatFeatureFlags): VkFormat =
    for format in canidates:
        var props: VkFormatProperties
        vkGetPhysicalDeviceFormatProperties(app.physicalDevice, format, addr props)
        if tiling == VK_IMAGE_TILING_LINEAR and (props.linearTilingFeatures.int and features.int) == features.int:
            return format
        elif tiling == VK_IMAGE_TILING_OPTIMAL and (props.optimalTilingFeatures.int and features.int) == features.int:
            return format

    raise newException(RuntimeException,"failed to find supported format!")

proc findDepthFormat(app: VulkanTutorialApp): VkFormat =
    return app.findSupportedFormat(@[VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT],VK_IMAGE_TILING_OPTIMAL,VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT.VkFormatFeatureFlags)

proc createDepthResources(app: VulkanTutorialApp) =
    var depthFormat: VkFormat = app.findDepthFormat()
    app.createImage(app.swapChainExtent.width, app.swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT.VkImageUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.VkMemoryPropertyFlags, app.depthImage, app.depthImageMemory);
    app.depthImageView = app.createImageView(app.depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT.VkImageAspectFlags);

    app.transitionImageLayout(app.depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)


proc findMemoryType(app: VulkanTutorialApp, typeFilter: uint32, properties: VkMemoryPropertyFlags) : uint32 =
    var memProperties: VkPhysicalDeviceMemoryProperties
    vkGetPhysicalDeviceMemoryProperties(app.physicalDevice, addr memProperties);
    for i in 0..<memProperties.memoryTypeCount:
        if (typeFilter and (1 shl i).uint32).bool and (memProperties.memoryTypes[i].propertyFlags.uint32 and properties.uint32) == properties.uint32:
            return i
    raise newException(RuntimeException, "failed to find suitable memory type!")

proc createBuffer(app: VulkanTutorialApp, size: VkDeviceSize, usage: VkBufferUsageFlagBits, properties: VkMemoryPropertyFlagBits, buffer: var VkBuffer, bufferMemory: var VkDeviceMemory) =
    let
        bufferInfo: VkBufferCreateInfo = VkBufferCreateInfo(
            sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size: size,
            usage: usage.VkBufferUsageFlags,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE
        )

    if vkCreateBuffer(app.device, addr bufferInfo, nil, addr buffer) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to create buffer!")
    var memRequirements: VkMemoryRequirements
    vkGetBufferMemoryRequirements(app.device, buffer, addr memRequirements)

    let allocInfo: VkMemoryAllocateInfo = newVkMemoryAllocateInfo(
        allocationSize = memRequirements.size,
        memoryTypeIndex = app.findMemoryType(memRequirements.memoryTypeBits.uint32, properties.VkMemoryPropertyFlags)
    )

    if vkAllocateMemory(app.device, addr allocInfo, nil, addr bufferMemory) != VK_SUCCESS: # There is a max memory allocation limit in vulkan. Should use an allocator either custom or VulkanMemoryAllocator. Should calculate offsets and use a single allocation
        raise newException(RuntimeException, "failed to allocate vertex buffer memory!");

    if vkBindBufferMemory(app.device, buffer, bufferMemory, 0.VkDeviceSize) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to bind buffer memory")


proc copyBuffer(app: VulkanTutorialApp, srcBuffer: VkBuffer, dstBuffer: var VkBuffer, size: VkDeviceSize) =
    var commandBuffer: VkCommandBuffer = app.beginSingleTimeCommands()

    let copyRegion: VkBufferCopy = newVkBufferCopy(
        srcOffset = 0.VkDeviceSize,
        dstOffset = 0.VkDeviceSize,
        size = size
    )
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, addr copyRegion)

    app.endSingleTimeCommands(commandBuffer)

proc loadModel(app: VulkanTutorialApp) =
    var roomLoader: ObjLoader = ObjLoader(file: open(MODEL_PATH))
    roomLoader.parseFile()
    if roomLoader.model.isNone:
        raise newException(RuntimeException,"Failed to Load model")
    let model = roomLoader.model.get
    #for tup in zip(model.geom_vertices,model.text_vertices):
    #    let (vert,text) = tup
    #    app.vertices.add(vertex(vert,vec3(1.0,1.0,1.0),vec2(text.x,1.0f - text.y)))
    var uniqueVertices: Table[Vertex,uint32]
    for mesh in model.meshes:
        for face in mesh.faces:
            for indx, vertIndx in face.vert_indices:
                var vertex = Vertex(color: vec3(1.0,1.0,1.0))
                vertex.pos = model.geom_vertices[vertIndx-1]
                vertex.texCoord = vec2(model.text_vertices[face.text_coords[indx]-1].x,1.0f-model.text_vertices[face.text_coords[indx]-1].y) # Fix

                if uniqueVertices.getOrDefault(vertex,0) == 0:
                    uniqueVertices[vertex] = app.vertices.high.uint32 + 1
                    app.vertices.add(vertex)

                app.sceneIndices.add(uniqueVertices[vertex])

proc createVertexBuffer(app: VulkanTutorialApp) =
    let bufferSize : uint32 = (sizeof(app.vertices[0]) * app.vertices.len).uint32
    var
        stagingBuffer: VkBuffer
        stagingBufferMemory: VkDeviceMemory

    app.createBuffer(bufferSize.VkDeviceSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory)

    var data : pointer
    if vkMapMemory(app.device, stagingBufferMemory, 0.VkDeviceSize, bufferSize.VkDeviceSize, 0.VkMemoryMapFlags, addr data) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to map memory")
    copyMem(data, addr app.vertices[0], bufferSize)
    let
        alignedSize : uint32 = (bufferSize - 1) - ((bufferSize - 1) mod app.deviceProperties.limits.nonCoherentAtomSize.uint32) + app.deviceProperties.limits.nonCoherentAtomSize.uint32
        vertexRange: VkMappedMemoryRange = newVkMappedMemoryRange(
            memory = stagingBufferMemory,
            offset = 0.VkDeviceSize,
            size   = alignedSize.VkDeviceSize
        )
    discard vkFlushMappedMemoryRanges(app.device, 1, addr vertexRange)
    vkUnmapMemory(app.device, stagingBufferMemory)

    app.createBuffer(bufferSize.VkDeviceSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT or VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, app.vertexBuffer, app.vertexBufferMemory)
    app.copyBuffer(stagingBuffer, app.vertexBuffer, bufferSize.VkDeviceSize)
    vkDestroyBuffer(app.device, stagingBuffer, nil)
    vkFreeMemory(app.device, stagingBufferMemory, nil)

proc createIndexBuffer(app: VulkanTutorialApp) =
    let bufferSize : uint32 = (sizeof(app.sceneIndices[0]) * app.sceneIndices.len).uint32
    var
        stagingBuffer: VkBuffer
        stagingBufferMemory: VkDeviceMemory

    app.createBuffer(bufferSize.VkDeviceSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory)

    var data : pointer
    if vkMapMemory(app.device, stagingBufferMemory, 0.VkDeviceSize, bufferSize.VkDeviceSize, 0.VkMemoryMapFlags, addr data) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to map memory")
    copyMem(data, addr app.sceneIndices[0], bufferSize)
    let
        alignedSize : uint32 = (bufferSize - 1) - ((bufferSize - 1) mod app.deviceProperties.limits.nonCoherentAtomSize.uint32) + app.deviceProperties.limits.nonCoherentAtomSize.uint32
        indexRange: VkMappedMemoryRange = newVkMappedMemoryRange(
            memory = stagingBufferMemory,
            offset = 0.VkDeviceSize,
            size   = alignedSize.VkDeviceSize
        )
    discard vkFlushMappedMemoryRanges(app.device, 1, addr indexRange)
    vkUnmapMemory(app.device, stagingBufferMemory)

    app.createBuffer(bufferSize.VkDeviceSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT or VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, app.indexBuffer, app.indexBufferMemory)
    app.copyBuffer(stagingBuffer, app.indexBuffer, bufferSize.VkDeviceSize)
    vkDestroyBuffer(app.device, stagingBuffer, nil)
    vkFreeMemory(app.device, stagingBufferMemory, nil)

proc createUniformBuffers(app: VulkanTutorialApp) =
    const bufferSize : uint32 = sizeof(UniformBufferObject).uint32

    app.uniformBuffers = newSeq[VkBuffer](MAX_FRAMES_IN_FLIGHT)
    app.uniformBuffersMemory = newSeq[VkDeviceMemory](MAX_FRAMES_IN_FLIGHT)
    app.uniformBuffersMapped = newSeq[pointer](MAX_FRAMES_IN_FLIGHT)

    for i in 0..<MAX_FRAMES_IN_FLIGHT:
        app.createBuffer(bufferSize.VkDeviceSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, app.uniformBuffers[i], app.uniformBuffersMemory[i])

        if vkMapMemory(app.device, app.uniformBuffersMemory[i], 0.VkDeviceSize, bufferSize.VkDeviceSize, 0.VkMemoryMapFlags, addr app.uniformBuffersMapped[i]) != VK_SUCCESS:
            raise newException(RuntimeException, "failed to map memory for " & $i & "uniform buffer")

proc createDescriptorPool(app: VulkanTutorialApp) =
    let
        poolSizes: array[2,VkDescriptorPoolSize] = [
            newVkDescriptorPoolSize(
                `type` = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount = MAX_FRAMES_IN_FLIGHT
            ),
            newVkDescriptorPoolSize(
                `type` = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount = MAX_FRAMES_IN_FLIGHT
            )
        ]
        poolInfo: VkDescriptorPoolCreateInfo = newVkDescriptorPoolCreateInfo(
            poolSizeCount = poolSizes.len.uint32,
            pPoolSizes = addr poolSizes[0],
            maxSets = MAX_FRAMES_IN_FLIGHT
        )
    if vkCreateDescriptorPool(app.device, addr poolInfo, nil, addr app.descriptorPool) != VK_SUCCESS:
        raise newException(RuntimeException,"failed to create descriptor pool!")

proc createDescriptorSets(app: VulkanTutorialApp) =
    var
        layouts: seq[VkDescriptorSetLayout] = newSeq[VkDescriptorSetLayout](MAX_FRAMES_IN_FLIGHT)
    for i in 0..<MAX_FRAMES_IN_FLIGHT:
        layouts[i] = app.descriptorSetLayout
    let allocInfo : VkDescriptorSetAllocateInfo = newVkDescriptorSetAllocateInfo(
        descriptorPool = app.descriptorPool,
        descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
        pSetLayouts = addr layouts[0]
    )
    app.descriptorSets = newSeq[VkDescriptorSet](MAX_FRAMES_IN_FLIGHT)
    if vkAllocateDescriptorSets(app.device, addr allocInfo, addr app.descriptorSets[0]) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to allocate descriptor sets!")
    for i in 0..<MAX_FRAMES_IN_FLIGHT:
        let
            bufferInfo : VkDescriptorBufferInfo = newVkDescriptorBufferInfo(
                buffer = app.uniformBuffers[i],
                offset = 0.VkDeviceSize,
                range = sizeof(UniformBufferObject).VkDeviceSize
            )
            imageInfo : VkDescriptorImageInfo = newVkDescriptorImageInfo(
                imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                imageView = app.textureImageView,
                sampler = app.textureSampler
            )
            descriptorWrites : array[2,VkWriteDescriptorSet] = [
                newVkWriteDescriptorSet(
                    dstSet = app.descriptorSets[i],
                    dstBinding = 0,
                    dstArrayElement = 0,
                    descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    descriptorCount = 1,
                    pBufferInfo = addr bufferInfo,
                    pImageInfo = nil,
                    pTexelBufferView = nil
                ),
                newVkWriteDescriptorSet(
                    dstSet = app.descriptorSets[i],
                    dstBinding = 1,
                    dstArrayElement = 0,
                    descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    descriptorCount = 1,
                    pBufferInfo = nil,
                    pImageInfo = addr imageInfo,
                    pTexelBufferView = nil
                )
            ]
        vkUpdateDescriptorSets(app.device, descriptorWrites.len.uint32, addr descriptorWrites[0], 0, nil)

proc createImage(app: VulkanTutorialApp, width: uint32, height: uint32, format: VkFormat, tiling: VkImageTiling, usage: VkImageUsageFlags, properties: VkMemoryPropertyFlags, image: var VkImage, imageMemory: var VkDeviceMemory): void =
    let imageInfo: VkImageCreateInfo = newVkImageCreateInfo(
            imageType = VK_IMAGE_TYPE_2D,
            extent = VkExtent3D(
                width: width,
                height: height,
                depth: 1
            ),
            mipLevels = 1,
            arrayLayers = 1,
            format = format,
            tiling = tiling,
            initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            usage = usage,
            sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            samples = VK_SAMPLE_COUNT_1_BIT,
            flags = 0.VkImageCreateFlags,
            queueFamilyIndexCount = 0,
            pQueueFamilyIndices = nil
        )

    if vkCreateImage(app.device, addr imageInfo, nil, addr image) != VK_SUCCESS:
        raise newException(RuntimeException,"failed to create image!")

    var memRequirements: VkMemoryRequirements
    vkGetImageMemoryRequirements(app.device, image, addr memRequirements)
    var allocInfo: VkMemoryAllocateInfo = newVkMemoryAllocateInfo(
        allocationSize = memRequirements.size,
        memoryTypeIndex = app.findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.VkMemoryPropertyFlags)
    )

    if vkAllocateMemory(app.device, addr allocInfo, nil, addr imageMemory) != VK_SUCCESS:
        raise newException(RuntimeException,"failed to allocate image memory")

    discard vkBindImageMemory(app.device, image, imageMemory, 0.VkDeviceSize)

proc beginSingleTimeCommands(app: VulkanTutorialApp): VkCommandBuffer =
    let allocInfo: VkCommandBufferAllocateInfo = newVkCommandBufferAllocateInfo(
        level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandPool = app.commandPool,
        commandBufferCount = 1
    )
    discard vkAllocateCommandBuffers(app.device, addr allocInfo, addr result)

    let beginInfo: VkCommandBufferBeginInfo = newVkCommandBufferBeginInfo(
        flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT.VkCommandBufferUsageFlags,
        pInheritanceInfo = nil
    )

    discard vkBeginCommandBuffer(result, addr beginInfo)

proc endSingleTimeCommands(app: VulkanTutorialApp, commandBuffer: VkCommandBuffer) =
    discard vkEndCommandBuffer(commandBuffer)
    let submitInfo: VkSubmitInfo = VkSubmitInfo(
        commandBufferCount: 1,
        pCommandBuffers: addr commandBuffer
    )
    discard vkQueueSubmit(app.graphicsQueue, 1.uint32, addr submitInfo, VK_NULL_HANDLE.VkFence)
    discard vkQueueWaitIdle(app.graphicsQueue)
    vkFreeCommandBuffers(app.device, app.commandPool, 1, addr commandBuffer)

proc transitionImageLayout(app: VulkanTutorialApp, image: VkImage, format: VkFormat, oldLayout: VkImageLayout, newlayout: VkImageLayout) =
    var
        commandBuffer: VkCommandBuffer = app.beginSingleTimeCommands()
        barrier: VkImageMemoryBarrier = newVkImageMemoryBarrier(
            oldlayout = oldlayout,
            newLayout = newLayout,
            srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            image = image,
            subresourceRange = newVkImageSubresourceRange(
                aspectMask = VK_IMAGE_ASPECT_COLOR_BIT.VkImageAspectFlags,
                baseMipLevel = 0,
                levelCount = 1,
                baseArrayLayer = 0,
                layerCount = 1
            ),
            srcAccessMask = 0.VkAccessFlags,
            dstAccessMask = 0.VkAccessFlags
        )
    var
        sourceStage: VkPipelineStageFlags
        destinationStage: VkPipelineStageFlags

    if newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT.VkImageAspectFlags

        if hasStencilComponent(format):
            barrier.subresourceRange.aspectMask = (barrier.subresourceRange.aspectMask.uint32 or VK_IMAGE_ASPECT_STENCIL_BIT.uint32).VkImageAspectFlags
    else:
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT.VkImageAspectFlags

    if oldLayout == VK_IMAGE_LAYOUT_UNDEFINED and newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        barrier.srcAccessMask = 0.VkAccessFlags
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT.VkAccessFlags

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT.VkPipelineStageFlags
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT.VkPipelineStageFlags
    elif oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT.VkAccessFlags
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT.VkAccessFlags

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT.VkPipelineStageFlags
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT.VkPipelineStageFlags
    elif oldLayout == VK_IMAGE_LAYOUT_UNDEFINED and newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        barrier.srcAccessMask = 0.VkAccessFlags
        barrier.dstAccessMask = (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT or VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT).VkAccessFlags

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT.VkPipelineStageFlags
        destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT.VkPipelineStageFlags
    else:
        raise newException(RuntimeException,"unsupported layout transition!")

    vkCmdPipelineBarrier(
        commandBuffer,
        0.VkPipelineStageFlags,
        0.VkPipelineStageFlags,
        0.VkDependencyFlags,
        0,
        nil,
        0,
        nil,
        1,
        addr barrier
    )
    app.endSingleTimeCommands(commandBuffer)

proc copyBufferToImage(app: VulkanTutorialApp, buffer: VkBuffer, image: VkImage, width: uint32, height: uint32) =
    var
        commandBuffer: VkCommandBuffer = app.beginSingleTimeCommands()
        region : VkBufferImageCopy = newVkBufferImageCopy(
            bufferOffset = 0.VkDeviceSize,
            bufferRowLength = 0,
            bufferImageHeight = 0,
            imageSubresource = newVkImageSubresourceLayers(
                aspectMask = VK_IMAGE_ASPECT_COLOR_BIT.VkImageAspectFlags,
                mipLevel = 0,
                baseArrayLayer = 0,
                layerCount = 1
            ),
            imageOffset = newVkOffset3D(0,0,0),
            imageExtent = VkExtent3D(
                width: width,
                height: height,
                depth: 1
            )
        )
    vkCmdCopyBufferToImage(
        commandBuffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        addr region
    )
    app.endSingleTimeCommands(commandBuffer)
#[ [TODO]
 All of the helper functions that submit commands so far have been set up to
 execute synchronously by waiting for the queue to become idle. For practical
 applications it is recommended to combine these operations in a single
 command buffer and execute them asynchronously for higher throughput,
 especially the transitions and copy in the createTextureImage function.
 Try to experiment with this by creating a setupCommandBuffer that the
 helper functions record commands into, and add a flushSetupCommands to
 execute the commands that have been recorded so far. It's best to do this
 after the texture mapping works to check if the texture resources are
 still set up correctly.
]#

proc createImageView(app: VulkanTutorialApp, image: VkImage, format: VkFormat, aspectFlags: VkImageAspectFlags): VkImageView =
    var viewInfo: VkImageViewCreateInfo = VkImageViewCreateInfo(
        image: image,
        viewType: VK_IMAGE_VIEW_TYPE_2D,
        format: format,
        subresourceRange: newVkImageSubresourceRange(
            aspectMask = aspectFlags,
            baseMipLevel = 0,
            levelCount = 1,
            baseArrayLayer = 0,
            layerCount = 1
        )
    )
    if vkCreateImageView(app.device, addr viewInfo, nil, addr result) != VK_SUCCESS:
        raise newException(RuntimeException,"failed to create texture image view!")


proc createTextureImageView(app: VulkanTutorialApp) =
    app.textureImageView = app.createImageView(app.textureImage,VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT.VkImageAspectFlags)

proc createTextureSampler(app: VulkanTutorialApp) =
    var samplerInfo: VkSamplerCreateInfo = newVkSamplerCreateInfo(
        magFilter = VK_FILTER_LINEAR,
        minFilter = VK_FILTER_LINEAR,
        addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        anisotropyEnable = VK_TRUE.VkBool32,
        maxAnisotropy = app.deviceProperties.limits.maxSamplerAnisotropy,
        borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        unnormalizedCoordinates = VK_FALSE.VkBool32,
        compareEnable = VK_FALSE.VkBool32,
        compareOp = VK_COMPARE_OP_ALWAYS,
        mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        mipLodBias = 0.0f,
        minLod = 0.0f,
        maxLod = 0.0f
    )
    if vkCreateSampler(app.device, addr samplerInfo, nil, addr app.textureSampler) != VK_SUCCESS:
        raise newException(RuntimeException,"failed to create texture sampler!")

proc createTextureImage(app: VulkanTutorialApp) =
    var
        texWidth : int
        texHeight : int
        texChannels : int
    const filename: cstring = cstring(TEXTURE_PATH)
    let
        pixels: cstring = stbi_load(filename, addr texWidth, addr texHeight, addr texChannels, STBI_rgb_alpha)
        imageSize: VkDeviceSize = (texWidth * texHeight * 4).VkDeviceSize
    if pixels.isNil:
        raise newException(RuntimeException,"failed to load texture image!")

    var
        stagingBuffer: VkBuffer
        stagingBufferMemory: VkDeviceMemory

    app.createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory)

    var data : pointer
    if vkMapMemory(app.device, stagingBufferMemory, 0.VkDeviceSize, imageSize, 0.VkMemoryMapFlags, addr data) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to map memory")
    copyMem(data, addr pixels[0], imageSize.int)
    let
        alignedSize : uint32 = (imageSize.uint32 - 1) - ((imageSize.uint32 - 1) mod app.deviceProperties.limits.nonCoherentAtomSize.uint32) + app.deviceProperties.limits.nonCoherentAtomSize.uint32
        vertexRange: VkMappedMemoryRange = newVkMappedMemoryRange(
            memory = stagingBufferMemory,
            offset = 0.VkDeviceSize,
            size   = alignedSize.VkDeviceSize
        )
    discard vkFlushMappedMemoryRanges(app.device, 1, addr vertexRange)
    vkUnmapMemory(app.device, stagingBufferMemory)
    stbi_image_free(pixels)
    app.createImage(texWidth.uint32,texHeight.uint32,VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, (VK_IMAGE_USAGE_TRANSFER_DST_BIT or VK_IMAGE_USAGE_SAMPLED_BIT).VkImageUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.VkMemoryPropertyFlags, app.textureImage, app.textureImageMemory)
    app.transitionImageLayout(app.textureImage, VK_FORMAT_B8G8R8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    app.copyBufferToImage(stagingBuffer, app.textureImage, texWidth.uint32, texHeight.uint32)
    app.transitionImageLayout(app.textureImage,VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)

    vkDestroyBuffer(app.device, stagingBuffer, nil)
    vkFreeMemory(app.device, stagingBufferMemory, nil)

proc createCommandBuffers(app: VulkanTutorialApp) =
    app.commandBuffers.setLen(MAX_FRAMES_IN_FLIGHT)
    let allocInfo: VkCommandBufferAllocateInfo = newVkCommandBufferAllocateInfo(
        commandPool = app.commandPool,
        level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount = cast[uint32](app.commandBuffers.len)
    )
    if vkAllocateCommandBuffers(app.device, addr allocInfo, addr app.commandBuffers[0]) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to allocate command buffers!")

proc recordCommandBuffer(app: VulkanTutorialApp, commandBuffer: var VkCommandBuffer, imageIndex: uint32) =
    let beginInfo: VkCommandBufferBeginInfo = newVkCommandBufferBeginInfo(
        flags = VkCommandBufferUsageFlags(0),
        pInheritanceInfo = nil # [TODO] I should really make this have a default nil value in nimvulkan
    )
    if vkBeginCommandBuffer(commandBuffer, addr beginInfo) != VK_SUCCESS:
        raise newException(RuntimeException, "failed to begin recording command buffer!")

    let
        clearValues : array[2,VkClearValue] = [
            VkClearValue(color: VkClearColorValue(float32: [0f, 0f, 0f, 1f])),
            VkClearValue(depthStencil: VkClearDepthStencilValue(depth: 1.0f, stencil: 0))
        ]
        renderPassInfo: VkRenderPassBeginInfo = newVkRenderPassBeginInfo(
            renderPass = app.renderPass,
            framebuffer = app.swapChainFrameBuffers[imageIndex],
            renderArea = VkRect2D(
                offset: VkOffset2d(x: 0,y: 0),
                extent: app.swapChainExtent
            ),
            clearValueCount = clearValues.len.uint32,
            pClearValues = addr clearValues[0]
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
    vkCmdBindIndexBuffer(commandBuffer, app.indexBuffer, 0.VkDeviceSize, VK_INDEX_TYPE_UINT32) # possible types are VK_INDEX_TYPE_UINT16 and VK_INDEX_TYPE_UINT32
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app.pipelineLayout, 0, 1, addr app.descriptorSets[app.currentFrame], 0, nil)
    vkCmdDrawIndexed(commandBuffer, app.sceneIndices.len.uint32, 1, 0, 0, 0)
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

proc updateUniformBuffer(app: VulkanTutorialApp, currentImage: uint32) =
    var
        currentTime = getMonoTime()
        time: float = (currentTime - startTime).inMilliseconds.float32
        ubo: UniformBufferObject = UniformBufferObject(
            model: rotate((time * toRadians(0.05f)).float32, vec3(0.0f,0.0f,1.0f)),
            view: lookAt(vec3(2.0f,2.0f,2.0f), vec3(0.0f,0.0f,0.0f), vec3(0.0f,0.0f,1.0f)), # Will be deprecated next vmath version use toAngles and figure out RH coord system
            proj: perspective[float32](45.0f, (app.swapChainExtent.width.float32 / app.swapChainExtent.height.float32), 0.1f, 10.0f),
        )
    ubo.proj = toVulY(ubo.proj)
    copyMem(app.uniformBuffersMapped[currentImage], addr ubo, sizeof(ubo))
    let
        alignedSize : uint32 = (sizeof(ubo) - 1) - ((sizeof(ubo) - 1) mod app.deviceProperties.limits.nonCoherentAtomSize.uint32) + app.deviceProperties.limits.nonCoherentAtomSize.uint32
        uniformRange: VkMappedMemoryRange = newVkMappedMemoryRange(
            memory = app.uniformBuffersMemory[currentImage],
            offset = 0.VkDeviceSize,
            size   = alignedSize.VkDeviceSize
        )
    discard vkFlushMappedMemoryRanges(app.device, 1, addr uniformRange)


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
    app.updateUniformBuffer(app.currentFrame)
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
    app.createDescriptorSetLayout()
    app.createGraphicsPipeline()
    app.createCommandPool()
    app.createDepthResources()
    app.createFrameBuffers()
    app.createTextureImage()
    app.createTextureImageView()
    app.createTextureSampler()
    app.loadModel()
    app.createVertexBuffer()
    app.createIndexBuffer()
    app.createUniformBuffers()
    app.createDescriptorPool()
    app.createDescriptorSets()
    app.createCommandBuffers()
    app.createSyncObjects()
    app.framebufferResized = false
    app.currentFrame = 0

proc mainLoop(app: VulkanTutorialApp) =
    while not glfw.windowShouldClose(app.window):
        glfw.pollEvents()
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
    vkDestroySampler(app.device, app.textureSampler, nil)
    vkDestroyImageView(app.device, app.textureImageView, nil)
    vkDestroyImage(app.device, app.textureImage, nil)
    vkFreeMemory(app.device, app.depthImageMemory, nil)
    vkFreeMemory(app.device, app.textureImageMemory, nil)
    for i in 0..<MAX_FRAMES_IN_FLIGHT:
        vkDestroyBuffer(app.device, app.uniformBuffers[i], nil)
        vkFreeMemory(app.device, app.uniformBuffersMemory[i], nil)
    vkDestroyDescriptorPool(app.device, app.descriptorPool, nil)
    vkDestroyDescriptorSetLayout(app.device, app.descriptorSetLayout, nil)
    vkDestroyBuffer(app.device, app.vertexBuffer, nil)
    vkFreeMemory(app.device, app.vertexBufferMemory, nil)
    vkDestroyBuffer(app.device, app.indexBuffer, nil);
    vkFreeMemory(app.device, app.indexBufferMemory, nil);
    vkDestroyDevice(app.device, nil) #destroy device before instance
    vkDestroySurfaceKHR(app.instance, app.surface, nil)
    vkDestroyInstance(app.instance, nil)
    glfw.destroyWindow(app.window)
    glfw.terminate()

proc run*(app: var VulkanTutorialApp) =
    app.initWindow()
    app.initVulkan()
    app.mainLoop()
    app.cleanup()
