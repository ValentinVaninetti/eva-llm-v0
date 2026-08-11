use std::os::raw::{c_char, c_void};

pub type VkInstance = *mut c_void;
pub type VkPhysicalDevice = *mut c_void;
pub type VkDevice = *mut c_void;
pub type VkQueue = *mut c_void;
pub type VkCommandBuffer = *mut c_void;
pub type VkBuffer = u64;
pub type VkDeviceMemory = u64;
pub type VkCommandPool = u64;
pub type VkDescriptorSet = u64;
pub type VkDescriptorSetLayout = u64;
pub type VkDescriptorPool = u64;
pub type VkPipeline = u64;
pub type VkPipelineLayout = u64;
pub type VkShaderModule = u64;
pub type VkFence = u64;
pub type VkDeviceSize = u64;
pub type VkFlags = u32;
pub type VkBool32 = u32;
pub type VkResult = i32;

pub const VK_SUCCESS: VkResult = 0;

// VkStructureType
pub const STRUCTURE_TYPE_APPLICATION_INFO: i32 = 0;
pub const STRUCTURE_TYPE_INSTANCE_CREATE_INFO: i32 = 1;
pub const STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: i32 = 2;
pub const STRUCTURE_TYPE_DEVICE_CREATE_INFO: i32 = 3;
pub const STRUCTURE_TYPE_SUBMIT_INFO: i32 = 4;
pub const STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: i32 = 5;
pub const STRUCTURE_TYPE_FENCE_CREATE_INFO: i32 = 8;
pub const STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: i32 = 18;
pub const STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: i32 = 16;
pub const STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: i32 = 29;
pub const STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: i32 = 30;
pub const STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: i32 = 32;
pub const STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: i32 = 33;
pub const STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: i32 = 34;
pub const STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: i32 = 35;
pub const STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: i32 = 39;
pub const STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: i32 = 40;
pub const STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: i32 = 42;
pub const STRUCTURE_TYPE_BUFFER_CREATE_INFO: i32 = 12;
pub const STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER: i32 = 44;

// VkQueueFlagBits
pub const QUEUE_COMPUTE_BIT: VkFlags = 0x00000002;

// VkBufferUsageFlagBits
pub const BUFFER_USAGE_TRANSFER_SRC_BIT: VkFlags = 0x00000001;
pub const BUFFER_USAGE_TRANSFER_DST_BIT: VkFlags = 0x00000002;
pub const BUFFER_USAGE_STORAGE_BUFFER_BIT: VkFlags = 0x00000020;

// VkMemoryPropertyFlagBits
pub const MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkFlags = 0x00000001;
pub const MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkFlags = 0x00000002;
pub const MEMORY_PROPERTY_HOST_COHERENT_BIT: VkFlags = 0x00000004;
pub const MEMORY_PROPERTY_HOST_CACHED_BIT: VkFlags = 0x00000008;

// VkDescriptorType
pub const DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;

// VkPipelineBindPoint
pub const PIPELINE_BIND_POINT_COMPUTE: u32 = 1;

// VkCommandBufferLevel
pub const COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;

// VkCommandBufferUsageFlagBits
pub const COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: VkFlags = 0x00000001;

// VkSharingMode
pub const SHARING_MODE_EXCLUSIVE: u32 = 0;

// VkShaderStageFlagBits
pub const SHADER_STAGE_COMPUTE_BIT: VkFlags = 0x00000020;

// VkPipelineStageFlagBits
pub const PIPELINE_STAGE_COMPUTE_SHADER_BIT: VkFlags = 0x00000800;
pub const PIPELINE_STAGE_TRANSFER_BIT: VkFlags = 0x00001000;
pub const PIPELINE_STAGE_HOST_BIT: VkFlags = 0x00004000;

// VkAccessFlagBits
pub const ACCESS_SHADER_READ_BIT: VkFlags = 0x00000020;
pub const ACCESS_SHADER_WRITE_BIT: VkFlags = 0x00000040;
pub const ACCESS_TRANSFER_READ_BIT: VkFlags = 0x00000800;
pub const ACCESS_TRANSFER_WRITE_BIT: VkFlags = 0x00001000;
pub const ACCESS_HOST_READ_BIT: VkFlags = 0x00002000;
pub const ACCESS_HOST_WRITE_BIT: VkFlags = 0x00004000;

// VkPhysicalDeviceType
pub const PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: i32 = 1;
pub const PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: i32 = 2;

pub const API_VERSION_1_0: u32 = 1 << 22;

pub const WHOLE_SIZE: u64 = u64::MAX;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkApplicationInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub p_application_name: *const c_char,
    pub application_version: u32,
    pub p_engine_name: *const c_char,
    pub engine_version: u32,
    pub api_version: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkInstanceCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub p_application_info: *const VkApplicationInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDeviceQueueCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub queue_family_index: u32,
    pub queue_count: u32,
    pub p_queue_priorities: *const f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDeviceCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub queue_create_info_count: u32,
    pub p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
    pub p_enabled_features: *const c_void,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkExtent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkQueueFamilyProperties {
    pub queue_flags: VkFlags,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: VkExtent3D,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPhysicalDeviceProperties {
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: i32,
    pub device_name: [c_char; 256],
    pub _pad: [u8; 2048],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkMemoryType {
    pub property_flags: VkFlags,
    pub heap_index: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkMemoryHeap {
    pub size: u64,
    pub flags: VkFlags,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPhysicalDeviceMemoryProperties {
    pub memory_type_count: u32,
    pub memory_types: [VkMemoryType; 32],
    pub memory_heap_count: u32,
    pub memory_heaps: [VkMemoryHeap; 16],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkBufferCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub size: VkDeviceSize,
    pub usage: VkFlags,
    pub sharing_mode: u32,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkMemoryAllocateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub allocation_size: VkDeviceSize,
    pub memory_type_index: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkMemoryRequirements {
    pub size: VkDeviceSize,
    pub alignment: VkDeviceSize,
    pub memory_type_bits: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkCommandPoolCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub queue_family_index: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkCommandBufferAllocateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub command_pool: VkCommandPool,
    pub level: u32,
    pub command_buffer_count: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkCommandBufferBeginInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub p_inheritance_info: *const c_void,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkSubmitInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub wait_semaphore_count: u32,
    pub p_wait_semaphores: *const u64,
    pub p_wait_dst_stage_mask: *const VkFlags,
    pub command_buffer_count: u32,
    pub p_command_buffers: *const VkCommandBuffer,
    pub signal_semaphore_count: u32,
    pub p_signal_semaphores: *const u64,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkFenceCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkBufferCopy {
    pub src_offset: VkDeviceSize,
    pub dst_offset: VkDeviceSize,
    pub size: VkDeviceSize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkShaderModuleCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub code_size: usize,
    pub p_code: *const u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: u32,
    pub descriptor_count: u32,
    pub stage_flags: VkFlags,
    pub p_immutable_samplers: *const c_void,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorSetLayoutCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub binding_count: u32,
    pub p_bindings: *const VkDescriptorSetLayoutBinding,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPushConstantRange {
    pub stage_flags: VkFlags,
    pub offset: u32,
    pub size: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineLayoutCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub set_layout_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
    pub push_constant_range_count: u32,
    pub p_push_constant_ranges: *const VkPushConstantRange,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorPoolSize {
    pub ty: u32,
    pub descriptor_count: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorPoolCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub max_sets: u32,
    pub pool_size_count: u32,
    pub p_pool_sizes: *const VkDescriptorPoolSize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorSetAllocateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub descriptor_pool: VkDescriptorPool,
    pub descriptor_set_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkDescriptorBufferInfo {
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub range: VkDeviceSize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkWriteDescriptorSet {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub dst_set: VkDescriptorSet,
    pub dst_binding: u32,
    pub dst_array_element: u32,
    pub descriptor_count: u32,
    pub descriptor_type: u32,
    pub p_image_info: *const c_void,
    pub p_buffer_info: *const VkDescriptorBufferInfo,
    pub p_texel_buffer_view: *const c_void,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkPipelineShaderStageCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub stage: VkFlags,
    pub module: VkShaderModule,
    pub p_name: *const c_char,
    pub p_specialization_info: *const c_void,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkComputePipelineCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: VkFlags,
    pub stage: VkPipelineShaderStageCreateInfo,
    pub layout: VkPipelineLayout,
    pub base_pipeline_handle: VkPipeline,
    pub base_pipeline_index: i32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct VkBufferMemoryBarrier {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub src_access_mask: VkFlags,
    pub dst_access_mask: VkFlags,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub size: VkDeviceSize,
}

#[link(name = "vulkan")]
extern "C" {
    pub fn vkCreateInstance(
        p_create_info: *const VkInstanceCreateInfo,
        p_allocator: *const c_void,
        p_instance: *mut VkInstance,
    ) -> VkResult;
    pub fn vkDestroyInstance(instance: VkInstance, p_allocator: *const c_void);
    pub fn vkEnumeratePhysicalDevices(
        instance: VkInstance,
        p_physical_device_count: *mut u32,
        p_physical_devices: *mut VkPhysicalDevice,
    ) -> VkResult;
    pub fn vkGetPhysicalDeviceProperties(
        physical_device: VkPhysicalDevice,
        p_properties: *mut VkPhysicalDeviceProperties,
    );
    pub fn vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device: VkPhysicalDevice,
        p_queue_family_property_count: *mut u32,
        p_queue_family_properties: *mut VkQueueFamilyProperties,
    );
    pub fn vkGetPhysicalDeviceMemoryProperties(
        physical_device: VkPhysicalDevice,
        p_memory_properties: *mut VkPhysicalDeviceMemoryProperties,
    );
    pub fn vkCreateDevice(
        physical_device: VkPhysicalDevice,
        p_create_info: *const VkDeviceCreateInfo,
        p_allocator: *const c_void,
        p_device: *mut VkDevice,
    ) -> VkResult;
    pub fn vkDestroyDevice(device: VkDevice, p_allocator: *const c_void);
    pub fn vkGetDeviceQueue(
        device: VkDevice,
        queue_family_index: u32,
        queue_index: u32,
        p_queue: *mut VkQueue,
    );
    pub fn vkDeviceWaitIdle(device: VkDevice) -> VkResult;
    pub fn vkQueueWaitIdle(queue: VkQueue) -> VkResult;
    pub fn vkCreateCommandPool(
        device: VkDevice,
        p_create_info: *const VkCommandPoolCreateInfo,
        p_allocator: *const c_void,
        p_command_pool: *mut VkCommandPool,
    ) -> VkResult;
    pub fn vkDestroyCommandPool(
        device: VkDevice,
        command_pool: VkCommandPool,
        p_allocator: *const c_void,
    );
    pub fn vkAllocateCommandBuffers(
        device: VkDevice,
        p_allocate_info: *const VkCommandBufferAllocateInfo,
        p_command_buffers: *mut VkCommandBuffer,
    ) -> VkResult;
    pub fn vkFreeCommandBuffers(
        device: VkDevice,
        command_pool: VkCommandPool,
        command_buffer_count: u32,
        p_command_buffers: *const VkCommandBuffer,
    );
    pub fn vkBeginCommandBuffer(
        command_buffer: VkCommandBuffer,
        p_begin_info: *const VkCommandBufferBeginInfo,
    ) -> VkResult;
    pub fn vkEndCommandBuffer(command_buffer: VkCommandBuffer) -> VkResult;
    pub fn vkQueueSubmit(
        queue: VkQueue,
        submit_count: u32,
        p_submits: *const VkSubmitInfo,
        fence: VkFence,
    ) -> VkResult;
    pub fn vkCreateShaderModule(
        device: VkDevice,
        p_create_info: *const VkShaderModuleCreateInfo,
        p_allocator: *const c_void,
        p_shader_module: *mut VkShaderModule,
    ) -> VkResult;
    pub fn vkDestroyShaderModule(
        device: VkDevice,
        shader_module: VkShaderModule,
        p_allocator: *const c_void,
    );
    pub fn vkCreateDescriptorSetLayout(
        device: VkDevice,
        p_create_info: *const VkDescriptorSetLayoutCreateInfo,
        p_allocator: *const c_void,
        p_set_layout: *mut VkDescriptorSetLayout,
    ) -> VkResult;
    pub fn vkDestroyDescriptorSetLayout(
        device: VkDevice,
        set_layout: VkDescriptorSetLayout,
        p_allocator: *const c_void,
    );
    pub fn vkCreatePipelineLayout(
        device: VkDevice,
        p_create_info: *const VkPipelineLayoutCreateInfo,
        p_allocator: *const c_void,
        p_pipeline_layout: *mut VkPipelineLayout,
    ) -> VkResult;
    pub fn vkDestroyPipelineLayout(
        device: VkDevice,
        pipeline_layout: VkPipelineLayout,
        p_allocator: *const c_void,
    );
    pub fn vkCreateComputePipelines(
        device: VkDevice,
        pipeline_cache: u64,
        create_info_count: u32,
        p_create_infos: *const VkComputePipelineCreateInfo,
        p_allocator: *const c_void,
        p_pipelines: *mut VkPipeline,
    ) -> VkResult;
    pub fn vkDestroyPipeline(device: VkDevice, pipeline: VkPipeline, p_allocator: *const c_void);
    pub fn vkCreateDescriptorPool(
        device: VkDevice,
        p_create_info: *const VkDescriptorPoolCreateInfo,
        p_allocator: *const c_void,
        p_descriptor_pool: *mut VkDescriptorPool,
    ) -> VkResult;
    pub fn vkDestroyDescriptorPool(
        device: VkDevice,
        descriptor_pool: VkDescriptorPool,
        p_allocator: *const c_void,
    );
    pub fn vkAllocateDescriptorSets(
        device: VkDevice,
        p_allocate_info: *const VkDescriptorSetAllocateInfo,
        p_descriptor_sets: *mut VkDescriptorSet,
    ) -> VkResult;
    pub fn vkUpdateDescriptorSets(
        device: VkDevice,
        descriptor_write_count: u32,
        p_descriptor_writes: *const VkWriteDescriptorSet,
        descriptor_copy_count: u32,
        p_descriptor_copies: *const c_void,
    );
    pub fn vkCreateBuffer(
        device: VkDevice,
        p_create_info: *const VkBufferCreateInfo,
        p_allocator: *const c_void,
        p_buffer: *mut VkBuffer,
    ) -> VkResult;
    pub fn vkDestroyBuffer(device: VkDevice, buffer: VkBuffer, p_allocator: *const c_void);
    pub fn vkGetBufferMemoryRequirements(
        device: VkDevice,
        buffer: VkBuffer,
        p_memory_requirements: *mut VkMemoryRequirements,
    );
    pub fn vkAllocateMemory(
        device: VkDevice,
        p_allocate_info: *const VkMemoryAllocateInfo,
        p_allocator: *const c_void,
        p_memory: *mut VkDeviceMemory,
    ) -> VkResult;
    pub fn vkFreeMemory(device: VkDevice, memory: VkDeviceMemory, p_allocator: *const c_void);
    pub fn vkBindBufferMemory(
        device: VkDevice,
        buffer: VkBuffer,
        memory: VkDeviceMemory,
        memory_offset: VkDeviceSize,
    ) -> VkResult;
    pub fn vkMapMemory(
        device: VkDevice,
        memory: VkDeviceMemory,
        offset: VkDeviceSize,
        size: VkDeviceSize,
        flags: VkFlags,
        pp_data: *mut *mut c_void,
    ) -> VkResult;
    pub fn vkUnmapMemory(device: VkDevice, memory: VkDeviceMemory);
    pub fn vkCreateFence(
        device: VkDevice,
        p_create_info: *const VkFenceCreateInfo,
        p_allocator: *const c_void,
        p_fence: *mut VkFence,
    ) -> VkResult;
    pub fn vkDestroyFence(device: VkDevice, fence: VkFence, p_allocator: *const c_void);
    pub fn vkResetFences(device: VkDevice, fence_count: u32, p_fences: *const VkFence) -> VkResult;
    pub fn vkWaitForFences(
        device: VkDevice,
        fence_count: u32,
        p_fences: *const VkFence,
        wait_all: VkBool32,
        timeout: u64,
    ) -> VkResult;
    pub fn vkCmdDispatch(
        command_buffer: VkCommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    );
    pub fn vkCmdBindPipeline(
        command_buffer: VkCommandBuffer,
        pipeline_bind_point: u32,
        pipeline: VkPipeline,
    );
    pub fn vkCmdBindDescriptorSets(
        command_buffer: VkCommandBuffer,
        pipeline_bind_point: u32,
        layout: VkPipelineLayout,
        first_set: u32,
        descriptor_set_count: u32,
        p_descriptor_sets: *const VkDescriptorSet,
        dynamic_offset_count: u32,
        p_dynamic_offsets: *const u32,
    );
    pub fn vkCmdPushConstants(
        command_buffer: VkCommandBuffer,
        layout: VkPipelineLayout,
        stage_flags: VkFlags,
        offset: u32,
        size: u32,
        p_values: *const c_void,
    );
    pub fn vkCmdPipelineBarrier(
        command_buffer: VkCommandBuffer,
        src_stage_mask: VkFlags,
        dst_stage_mask: VkFlags,
        dependency_flags: VkFlags,
        memory_barrier_count: u32,
        p_memory_barriers: *const c_void,
        buffer_memory_barrier_count: u32,
        p_buffer_memory_barriers: *const VkBufferMemoryBarrier,
        image_memory_barrier_count: u32,
        p_image_memory_barriers: *const c_void,
    );
    pub fn vkCmdCopyBuffer(
        command_buffer: VkCommandBuffer,
        src_buffer: VkBuffer,
        dst_buffer: VkBuffer,
        region_count: u32,
        p_regions: *const VkBufferCopy,
    );
}
