pub mod ffi;

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::time::Instant;

use ffi::*;

const MATMUL_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul.spv"));

pub struct GpuInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: i32,
    pub api_version: u32,
}

pub struct Gpu {
    instance: VkInstance,
    device: VkDevice,
    queue: VkQueue,
    cmd_pool: VkCommandPool,
    cmd: VkCommandBuffer,
    fence: VkFence,
    mem_props: VkPhysicalDeviceMemoryProperties,
    info: GpuInfo,
    shader: VkShaderModule,
    ds_layout: VkDescriptorSetLayout,
    pipeline_layout: VkPipelineLayout,
    pipeline: VkPipeline,
    desc_pool: VkDescriptorPool,
}

struct Buffer {
    device: VkDevice,
    buffer: VkBuffer,
    memory: VkDeviceMemory,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            vkDestroyBuffer(self.device, self.buffer, ptr::null());
            vkFreeMemory(self.device, self.memory, ptr::null());
        }
    }
}

impl Gpu {
    pub fn init() -> Result<Gpu, String> {
        unsafe {
            let app = VkApplicationInfo {
                s_type: STRUCTURE_TYPE_APPLICATION_INFO,
                p_next: ptr::null(),
                p_application_name: c"eva".as_ptr(),
                application_version: 0,
                p_engine_name: ptr::null(),
                engine_version: 0,
                api_version: API_VERSION_1_0,
            };
            let inst_info = VkInstanceCreateInfo {
                s_type: STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                p_application_info: &app,
                enabled_layer_count: 0,
                pp_enabled_layer_names: ptr::null(),
                enabled_extension_count: 0,
                pp_enabled_extension_names: ptr::null(),
            };
            let mut instance: VkInstance = ptr::null_mut();
            check(vkCreateInstance(&inst_info, ptr::null(), &mut instance), "vkCreateInstance")?;

            let mut count = 0u32;
            check(
                vkEnumeratePhysicalDevices(instance, &mut count, ptr::null_mut()),
                "vkEnumeratePhysicalDevices",
            )?;
            if count == 0 {
                return Err("no hay ninguna GPU Vulkan en este sistema".to_string());
            }
            let mut devices = vec![ptr::null_mut(); count as usize];
            check(
                vkEnumeratePhysicalDevices(instance, &mut count, devices.as_mut_ptr()),
                "vkEnumeratePhysicalDevices",
            )?;

            let mut best: Option<(VkPhysicalDevice, i32)> = None;
            let mut best_queue = 0u32;
            for &pd in &devices {
                let props = physical_props(pd);
                let qfam = queue_family(pd).ok_or_else(|| "sin cola de cómputo".to_string())?;
                let score = match props.device_type {
                    PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => 2,
                    PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => 1,
                    _ => 0,
                };
                if best.map(|(_, s)| score > s).unwrap_or(true) {
                    best = Some((pd, score));
                    best_queue = qfam;
                }
            }
            let (physical, _) = best.ok_or("no se encontró GPU con cola de cómputo")?;
            let props = physical_props(physical);

            let mut mem_props = VkPhysicalDeviceMemoryProperties {
                memory_type_count: 0,
                memory_types: [VkMemoryType { property_flags: 0, heap_index: 0 }; 32],
                memory_heap_count: 0,
                memory_heaps: [VkMemoryHeap { size: 0, flags: 0 }; 16],
            };
            vkGetPhysicalDeviceMemoryProperties(physical, &mut mem_props);

            let priority = 1.0f32;
            let qci = VkDeviceQueueCreateInfo {
                s_type: STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                queue_family_index: best_queue,
                queue_count: 1,
                p_queue_priorities: &priority,
            };
            let dev_info = VkDeviceCreateInfo {
                s_type: STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                queue_create_info_count: 1,
                p_queue_create_infos: &qci,
                enabled_layer_count: 0,
                pp_enabled_layer_names: ptr::null(),
                enabled_extension_count: 0,
                pp_enabled_extension_names: ptr::null(),
                p_enabled_features: ptr::null(),
            };
            let mut device: VkDevice = ptr::null_mut();
            check(vkCreateDevice(physical, &dev_info, ptr::null(), &mut device), "vkCreateDevice")?;

            let mut queue: VkQueue = ptr::null_mut();
            vkGetDeviceQueue(device, best_queue, 0, &mut queue);

            let cpci = VkCommandPoolCreateInfo {
                s_type: STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                queue_family_index: best_queue,
            };
            let mut cmd_pool: VkCommandPool = 0;
            check(vkCreateCommandPool(device, &cpci, ptr::null(), &mut cmd_pool), "vkCreateCommandPool")?;

            let caci = VkCommandBufferAllocateInfo {
                s_type: STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: ptr::null(),
                command_pool: cmd_pool,
                level: COMMAND_BUFFER_LEVEL_PRIMARY,
                command_buffer_count: 1,
            };
            let mut cmd: VkCommandBuffer = ptr::null_mut();
            check(vkAllocateCommandBuffers(device, &caci, &mut cmd), "vkAllocateCommandBuffers")?;

            let fci = VkFenceCreateInfo {
                s_type: STRUCTURE_TYPE_FENCE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
            };
            let mut fence: VkFence = 0;
            check(vkCreateFence(device, &fci, ptr::null(), &mut fence), "vkCreateFence")?;

            let info = GpuInfo {
                name: CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().to_string(),
                vendor_id: props.vendor_id,
                device_id: props.device_id,
                device_type: props.device_type,
                api_version: props.api_version,
            };

            let shader = create_shader(device)?;
            let ds_layout = create_ds_layout(device)?;
            let pipeline_layout = create_pipeline_layout(device, ds_layout)?;
            let pipeline = create_pipeline(device, shader, pipeline_layout)?;
            let desc_pool = create_desc_pool(device)?;

            Ok(Gpu {
                instance,
                device,
                queue,
                cmd_pool,
                cmd,
                fence,
                mem_props,
                info,
                shader,
                ds_layout,
                pipeline_layout,
                pipeline,
                desc_pool,
            })
        }
    }

    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, String> {
        let t0 = Instant::now();
        let sa = self.host_buffer(a.len() * 4)?;
        let sb = self.host_buffer(b.len() * 4)?;
        let sc = self.host_buffer(m * n * 4)?;
        self.write(&sa, a);
        self.write(&sb, b);

        let da = self.device_buffer(a.len() * 4, BUFFER_USAGE_TRANSFER_DST_BIT | BUFFER_USAGE_STORAGE_BUFFER_BIT)?;
        let db = self.device_buffer(b.len() * 4, BUFFER_USAGE_TRANSFER_DST_BIT | BUFFER_USAGE_STORAGE_BUFFER_BIT)?;
        let dc = self.device_buffer(m * n * 4, BUFFER_USAGE_TRANSFER_SRC_BIT | BUFFER_USAGE_STORAGE_BUFFER_BIT)?;

        let ds = self.alloc_ds()?;
        self.bind_ds(ds, &da, &db, &dc)?;

        self.record(|cb| {
            unsafe {
                // host -> staging (coherent) -> copy to device
                self.barrier(
                    cb,
                    PIPELINE_STAGE_HOST_BIT,
                    PIPELINE_STAGE_TRANSFER_BIT,
                    ACCESS_HOST_WRITE_BIT,
                    ACCESS_TRANSFER_READ_BIT,
                    &[&sa, &sb],
                );
                let region = VkBufferCopy { src_offset: 0, dst_offset: 0, size: (a.len() * 4) as u64 };
                vkCmdCopyBuffer(cb, sa.buffer, da.buffer, 1, &region);
                let region = VkBufferCopy { src_offset: 0, dst_offset: 0, size: (b.len() * 4) as u64 };
                vkCmdCopyBuffer(cb, sb.buffer, db.buffer, 1, &region);

                // transfer -> compute
                self.barrier(
                    cb,
                    PIPELINE_STAGE_TRANSFER_BIT,
                    PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    ACCESS_TRANSFER_WRITE_BIT,
                    ACCESS_SHADER_READ_BIT,
                    &[&da, &db],
                );

                vkCmdBindPipeline(cb, PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
                vkCmdBindDescriptorSets(
                    cb,
                    PIPELINE_BIND_POINT_COMPUTE,
                    self.pipeline_layout,
                    0,
                    1,
                    &ds,
                    0,
                    ptr::null(),
                );
                let params = [m as u32, k as u32, n as u32];
                vkCmdPushConstants(cb, self.pipeline_layout, SHADER_STAGE_COMPUTE_BIT, 0, 12, params.as_ptr().cast());
                vkCmdDispatch(cb, ((n + 15) / 16) as u32, ((m + 15) / 16) as u32, 1);

                // compute -> transfer (read C back)
                self.barrier(
                    cb,
                    PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    PIPELINE_STAGE_TRANSFER_BIT,
                    ACCESS_SHADER_WRITE_BIT,
                    ACCESS_TRANSFER_READ_BIT,
                    &[&dc],
                );
                let region = VkBufferCopy { src_offset: 0, dst_offset: 0, size: (m * n * 4) as u64 };
                vkCmdCopyBuffer(cb, dc.buffer, sc.buffer, 1, &region);

                // transfer -> host
                self.barrier(
                    cb,
                    PIPELINE_STAGE_TRANSFER_BIT,
                    PIPELINE_STAGE_HOST_BIT,
                    ACCESS_TRANSFER_WRITE_BIT,
                    ACCESS_HOST_READ_BIT,
                    &[&sc],
                );
            }
        })?;

        let out = self.read(&sc, m * n);
        let t = t0.elapsed();
        eprintln!("eva gpu: matmul {}x{}x{} en {:.3}ms", m, k, n, t.as_secs_f64() * 1e3);
        out
    }

    fn record(&self, f: impl FnOnce(VkCommandBuffer)) -> Result<(), String> {
        unsafe {
            let begin = VkCommandBufferBeginInfo {
                s_type: STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                p_inheritance_info: ptr::null(),
            };
            check(vkBeginCommandBuffer(self.cmd, &begin), "vkBeginCommandBuffer")?;
            f(self.cmd);
            check(vkEndCommandBuffer(self.cmd), "vkEndCommandBuffer")?;

            let submit = VkSubmitInfo {
                s_type: STRUCTURE_TYPE_SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &self.cmd,
                signal_semaphore_count: 0,
                p_signal_semaphores: ptr::null(),
            };
            check(vkResetFences(self.device, 1, &self.fence), "vkResetFences")?;
            check(vkQueueSubmit(self.queue, 1, &submit, self.fence), "vkQueueSubmit")?;
            check(
                vkWaitForFences(self.device, 1, &self.fence, 1, u64::MAX),
                "vkWaitForFences",
            )?;
        }
        Ok(())
    }

    fn barrier(
        &self,
        cb: VkCommandBuffer,
        src_stage: VkFlags,
        dst_stage: VkFlags,
        src_access: VkFlags,
        dst_access: VkFlags,
        bufs: &[&Buffer],
    ) {
        let barriers: Vec<VkBufferMemoryBarrier> = bufs
            .iter()
            .map(|b| VkBufferMemoryBarrier {
                s_type: STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: src_access,
                dst_access_mask: dst_access,
                src_queue_family_index: u32::MAX,
                dst_queue_family_index: u32::MAX,
                buffer: b.buffer,
                offset: 0,
                size: WHOLE_SIZE,
            })
            .collect();
        unsafe {
            vkCmdPipelineBarrier(
                cb,
                src_stage,
                dst_stage,
                0,
                0,
                ptr::null(),
                barriers.len() as u32,
                barriers.as_ptr(),
                0,
                ptr::null(),
            );
        }
    }

    fn device_buffer(&self, size: usize, usage: VkFlags) -> Result<Buffer, String> {
        self.buffer(size, usage, MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    }

    fn host_buffer(&self, size: usize) -> Result<Buffer, String> {
        self.buffer(
            size,
            BUFFER_USAGE_TRANSFER_SRC_BIT | BUFFER_USAGE_TRANSFER_DST_BIT,
            MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
    }

    fn buffer(&self, size: usize, usage: VkFlags, want: VkFlags) -> Result<Buffer, String> {
        unsafe {
            let ci = VkBufferCreateInfo {
                s_type: STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                size: size as u64,
                usage,
                sharing_mode: SHARING_MODE_EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: ptr::null(),
            };
            let mut buffer: VkBuffer = 0;
            check(vkCreateBuffer(self.device, &ci, ptr::null(), &mut buffer), "vkCreateBuffer")?;

            let mut reqs = VkMemoryRequirements { size: 0, alignment: 0, memory_type_bits: 0 };
            vkGetBufferMemoryRequirements(self.device, buffer, &mut reqs);

            let idx = self
                .find_memory_type(reqs.memory_type_bits, want)
                .ok_or("no hay tipo de memoria que cumpla los requisitos")?;
            let ai = VkMemoryAllocateInfo {
                s_type: STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                p_next: ptr::null(),
                allocation_size: reqs.size,
                memory_type_index: idx,
            };
            let mut memory: VkDeviceMemory = 0;
            check(vkAllocateMemory(self.device, &ai, ptr::null(), &mut memory), "vkAllocateMemory")?;
            check(vkBindBufferMemory(self.device, buffer, memory, 0), "vkBindBufferMemory")?;

            Ok(Buffer { device: self.device, buffer, memory })
        }
    }

    fn find_memory_type(&self, type_bits: u32, want: VkFlags) -> Option<u32> {
        for i in 0..self.mem_props.memory_type_count as usize {
            if (type_bits >> i) & 1 == 1
                && self.mem_props.memory_types[i].property_flags & want == want
            {
                return Some(i as u32);
            }
        }
        None
    }

    fn write(&self, b: &Buffer, data: &[f32]) {
        unsafe {
            let mut ptr: *mut c_void = ptr::null_mut();
            vkMapMemory(self.device, b.memory, 0, (data.len() * 4) as u64, 0, &mut ptr);
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<f32>(), data.len());
            vkUnmapMemory(self.device, b.memory);
        }
    }

    fn read(&self, b: &Buffer, len: usize) -> Result<Vec<f32>, String> {
        unsafe {
            let mut ptr: *mut c_void = ptr::null_mut();
            check(
                vkMapMemory(self.device, b.memory, 0, (len * 4) as u64, 0, &mut ptr),
                "vkMapMemory",
            )?;
            let mut out = vec![0.0f32; len];
            std::ptr::copy_nonoverlapping(ptr.cast::<f32>(), out.as_mut_ptr(), len);
            vkUnmapMemory(self.device, b.memory);
            Ok(out)
        }
    }

    fn alloc_ds(&self) -> Result<VkDescriptorSet, String> {
        unsafe {
            let ai = VkDescriptorSetAllocateInfo {
                s_type: STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: ptr::null(),
                descriptor_pool: self.desc_pool,
                descriptor_set_count: 1,
                p_set_layouts: &self.ds_layout,
            };
            let mut ds: VkDescriptorSet = 0;
            check(vkAllocateDescriptorSets(self.device, &ai, &mut ds), "vkAllocateDescriptorSets")?;
            Ok(ds)
        }
    }

    fn bind_ds(&self, ds: VkDescriptorSet, a: &Buffer, b: &Buffer, c: &Buffer) -> Result<(), String> {
        unsafe {
            let infos = [
                VkDescriptorBufferInfo { buffer: a.buffer, offset: 0, range: WHOLE_SIZE },
                VkDescriptorBufferInfo { buffer: b.buffer, offset: 0, range: WHOLE_SIZE },
                VkDescriptorBufferInfo { buffer: c.buffer, offset: 0, range: WHOLE_SIZE },
            ];
            let writes: Vec<VkWriteDescriptorSet> = (0..3)
                .map(|i| VkWriteDescriptorSet {
                    s_type: STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: ds,
                    dst_binding: i as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    p_image_info: ptr::null(),
                    p_buffer_info: &infos[i],
                    p_texel_buffer_view: ptr::null(),
                })
                .collect();
            vkUpdateDescriptorSets(self.device, 3, writes.as_ptr(), 0, ptr::null());
        }
        Ok(())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            vkDeviceWaitIdle(self.device);
            vkDestroyPipeline(self.device, self.pipeline, ptr::null());
            vkDestroyPipelineLayout(self.device, self.pipeline_layout, ptr::null());
            vkDestroyDescriptorSetLayout(self.device, self.ds_layout, ptr::null());
            vkDestroyDescriptorPool(self.device, self.desc_pool, ptr::null());
            vkDestroyShaderModule(self.device, self.shader, ptr::null());
            vkDestroyFence(self.device, self.fence, ptr::null());
            vkFreeCommandBuffers(self.device, self.cmd_pool, 1, &self.cmd);
            vkDestroyCommandPool(self.device, self.cmd_pool, ptr::null());
            vkDestroyDevice(self.device, ptr::null());
            vkDestroyInstance(self.instance, ptr::null());
        }
    }
}

fn physical_props(pd: VkPhysicalDevice) -> VkPhysicalDeviceProperties {
    unsafe {
        let mut props = VkPhysicalDeviceProperties {
            api_version: 0,
            driver_version: 0,
            vendor_id: 0,
            device_id: 0,
            device_type: 0,
            device_name: [0 as c_char; 256],
            _pad: [0; 2048],
        };
        vkGetPhysicalDeviceProperties(pd, &mut props);
        props
    }
}

fn queue_family(pd: VkPhysicalDevice) -> Option<u32> {
    unsafe {
        let mut count = 0u32;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &mut count, ptr::null_mut());
        let mut props = vec![
            VkQueueFamilyProperties {
                queue_flags: 0,
                queue_count: 0,
                timestamp_valid_bits: 0,
                min_image_transfer_granularity: VkExtent3D { width: 0, height: 0, depth: 0 },
            };
            count as usize
        ];
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &mut count, props.as_mut_ptr());
        for (i, p) in props.iter().enumerate() {
            if p.queue_flags & QUEUE_COMPUTE_BIT != 0 && p.queue_count > 0 {
                return Some(i as u32);
            }
        }
        None
    }
}

fn create_shader(device: VkDevice) -> Result<VkShaderModule, String> {
    unsafe {
        let ci = VkShaderModuleCreateInfo {
            s_type: STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            code_size: MATMUL_SPV.len(),
            p_code: MATMUL_SPV.as_ptr().cast(),
        };
        let mut m: VkShaderModule = 0;
        check(vkCreateShaderModule(device, &ci, ptr::null(), &mut m), "vkCreateShaderModule")?;
        Ok(m)
    }
}

fn create_ds_layout(device: VkDevice) -> Result<VkDescriptorSetLayout, String> {
    unsafe {
        let bindings = [
            VkDescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            },
            VkDescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            },
            VkDescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            },
        ];
        let ci = VkDescriptorSetLayoutCreateInfo {
            s_type: STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            binding_count: 3,
            p_bindings: bindings.as_ptr(),
        };
        let mut l: VkDescriptorSetLayout = 0;
        check(vkCreateDescriptorSetLayout(device, &ci, ptr::null(), &mut l), "vkCreateDescriptorSetLayout")?;
        Ok(l)
    }
}

fn create_pipeline_layout(device: VkDevice, ds_layout: VkDescriptorSetLayout) -> Result<VkPipelineLayout, String> {
    unsafe {
        let pc = VkPushConstantRange {
            stage_flags: SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: 12,
        };
        let ci = VkPipelineLayoutCreateInfo {
            s_type: STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &ds_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &pc,
        };
        let mut l: VkPipelineLayout = 0;
        check(vkCreatePipelineLayout(device, &ci, ptr::null(), &mut l), "vkCreatePipelineLayout")?;
        Ok(l)
    }
}

fn create_pipeline(
    device: VkDevice,
    shader: VkShaderModule,
    layout: VkPipelineLayout,
) -> Result<VkPipeline, String> {
    unsafe {
        let entry = c"main";
        let stage = VkPipelineShaderStageCreateInfo {
            s_type: STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage: SHADER_STAGE_COMPUTE_BIT,
            module: shader,
            p_name: entry.as_ptr(),
            p_specialization_info: ptr::null(),
        };
        let ci = VkComputePipelineCreateInfo {
            s_type: STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage,
            layout,
            base_pipeline_handle: 0,
            base_pipeline_index: -1,
        };
        let mut p: VkPipeline = 0;
        check(
            vkCreateComputePipelines(device, 0, 1, &ci, ptr::null(), &mut p),
            "vkCreateComputePipelines",
        )?;
        Ok(p)
    }
}

fn create_desc_pool(device: VkDevice) -> Result<VkDescriptorPool, String> {
    unsafe {
        let size = VkDescriptorPoolSize {
            ty: DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: 3,
        };
        let ci = VkDescriptorPoolCreateInfo {
            s_type: STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &size,
        };
        let mut p: VkDescriptorPool = 0;
        check(vkCreateDescriptorPool(device, &ci, ptr::null(), &mut p), "vkCreateDescriptorPool")?;
        Ok(p)
    }
}

fn check(result: VkResult, what: &str) -> Result<(), String> {
    if result == VK_SUCCESS {
        Ok(())
    } else {
        Err(format!("{} falló con VkResult {}", what, result))
    }
}
