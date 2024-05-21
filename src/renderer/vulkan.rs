use std::cmp;
use std::mem;
use std::sync::Mutex;
use std::ffi::{CStr, CString};

use winit::window::Window;
use taffy::{Size, AvailableSpace};
use ash::{vk, khr, Entry, Instance, Device};
use raw_window_handle::{HasWindowHandle, HasDisplayHandle, RawWindowHandle, RawDisplayHandle};

use crate::Dom;

const SHADER_CODE: &[u8] = include_bytes!("shader.spv");

// Height/width of the compute shader workgroup
const WG_SIZE: u32 = 8;

#[repr(packed)]
struct Box {
    pub pos: [f32; 2],
    pub size: [f32; 2],
    pub color: [f32; 4],
    pub radius: f32,
    pub padding: [f32; 3]
}

#[repr(packed)]
struct BoxBufferContents {
    len: u32,
    padding: [u32; 3],
    boxes: [Box; BOX_LIST_LEN]
}

const BOX_LIST_LEN: usize = 255;
const BOX_BUFFER_SIZE: usize = mem::size_of::<BoxBufferContents>();

struct BoxBuffer {
    buf: (vk::Buffer, vk::DeviceMemory, *mut u8),
    staging_buf: Option<(vk::Buffer, vk::DeviceMemory, *mut u8)>
}

unsafe impl Send for BoxBuffer {}

/// Synchronization objects for a frame
struct SyncSet {
    swap_image_avail: vk::Semaphore,
    cmd_buf_done: vk::Semaphore,
    frame_done: vk::Fence
}

pub struct Renderer {
    frame_idx: usize,
    box_bufs: Vec<BoxBuffer>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    desc_sets: Vec<vk::DescriptorSet>,
    desc_pool: vk::DescriptorPool,
    cmd_bufs: Vec<vk::CommandBuffer>,
    cmd_pool: vk::CommandPool,
    sync_sets: Vec<SyncSet>,
    swapchain_image_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    swapchain_ext: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    device: Device,
    gfx_queue: vk::Queue,
    surface: vk::SurfaceKHR,
    instance: Instance,
    _entry: Entry
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let window_handle = window.window_handle().unwrap().as_raw();
        let display_handle = window.display_handle().unwrap().as_raw();

        // Load vulkan
        let entry = unsafe { Entry::load().unwrap() };

        // Required instance extensions
        let mut req_instance_exts = vec![khr::surface::NAME.as_ptr()];

        match window_handle {
            RawWindowHandle::Xcb(_) => req_instance_exts.push(khr::xcb_surface::NAME.as_ptr()),
            RawWindowHandle::Xlib(_) => req_instance_exts.push(khr::xlib_surface::NAME.as_ptr()),
            RawWindowHandle::Win32(_) => req_instance_exts.push(khr::win32_surface::NAME.as_ptr()),
            _ => unimplemented!()
        }

        // Check if required instance extensions are available
        let avail_exts = unsafe { entry.enumerate_instance_extension_properties(None).unwrap() };

        for &req_ext in &req_instance_exts {
            let req_ext_name = unsafe { CStr::from_ptr(req_ext) };

            let found = avail_exts
                .iter()
                .map(|ext| ext.extension_name_as_c_str().unwrap())
                .any(|ext_name| ext_name == req_ext_name);
    
            if !found {
                panic!("Required instance extension {req_ext_name:?} not available");
            }
        }

        // Required instance layers
        let validation_layer_name = CString::new("VK_LAYER_KHRONOS_validation").unwrap();

        let mut req_layers = vec![];

        if cfg!(debug_assertions) {
            req_layers.push(validation_layer_name.as_ptr());
        }

        // Check if required instance layers are available
        let avail_layers = unsafe { entry.enumerate_instance_layer_properties().unwrap() };

        for &req_layer in &req_layers {
            let req_layer_name = unsafe { CStr::from_ptr(req_layer) };

            let found = avail_layers
                .iter()
                .map(|layer| layer.layer_name_as_c_str().unwrap())
                .any(|layer_name| layer_name == req_layer_name);
    
            if !found {
                panic!("Required instance extension {req_layer_name:?} not available");
            }
        }

        // Create instance
        let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 0, 0));

        let create_info = vk::InstanceCreateInfo::default()
            .enabled_extension_names(&req_instance_exts)
            .enabled_layer_names(&req_layers);

        let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };
        let surface_ext = khr::surface::Instance::new(&entry, &instance);

        // Create surface
        let surface = match (window_handle, display_handle) {
            (RawWindowHandle::Xcb(window_handle), RawDisplayHandle::Xcb(display_handle)) => {
                let ext = khr::xcb_surface::Instance::new(&entry, &instance);

                let create_info = vk::XcbSurfaceCreateInfoKHR::default()
                    .connection(display_handle.connection.unwrap().as_ptr())
                    .window(window_handle.window.get());

                unsafe { ext.create_xcb_surface(&create_info, None).unwrap() }
            },

            (RawWindowHandle::Xlib(window_handle), RawDisplayHandle::Xlib(display_handle)) => {
                let ext = khr::xlib_surface::Instance::new(&entry, &instance);

                let create_info = vk::XlibSurfaceCreateInfoKHR::default()
                    .dpy(display_handle.display.unwrap().as_ptr())
                    .window(window_handle.window);

                unsafe { ext.create_xlib_surface(&create_info, None).unwrap() }
            },

            (RawWindowHandle::Win32(window_handle), RawDisplayHandle::Windows(_)) => {
                let ext = khr::win32_surface::Instance::new(&entry, &instance);

                let create_info = vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(window_handle.hinstance.unwrap().get())
                    .hwnd(window_handle.hwnd.get());

                unsafe { ext.create_win32_surface(&create_info, None).unwrap() }
            },

            _ => unimplemented!()
        };

        // Required device extensions
        let req_device_exts = [khr::swapchain::NAME.as_ptr()];

        // Get available devices
        let phys_devs = unsafe { instance.enumerate_physical_devices().unwrap() };

        // Shortlist eligible devices
        struct EligibleDevice {
            phys_dev: vk::PhysicalDevice,
            gfx_queue_family: u32,
            props: vk::PhysicalDeviceProperties
        }

        // Check if the device has a queue family supporting graphics operations
        let has_gfx_queue = |&phys_dev| {
            let queue_props = unsafe { instance.get_physical_device_queue_family_properties(phys_dev) };

            queue_props
                .iter()
                .enumerate()
                .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|(queue_family, _)| (phys_dev, queue_family as u32))
        };

        // Check if the device and graphics queue family support the surface
        let supports_surface = |info: &(vk::PhysicalDevice, u32)| unsafe {
            let (phys_dev, gfx_queue_family) = *info;

            surface_ext
                .get_physical_device_surface_support(phys_dev, gfx_queue_family, surface)
                .eq(&Ok(true))
        };

        // Check if the device supports needed extensions
        let supports_extensions = |info: &(vk::PhysicalDevice, u32)| unsafe {
            let phys_dev = info.0;

            instance
                .enumerate_device_extension_properties(phys_dev)
                .map(|avail_exts| {
                    for req_ext in req_device_exts {
                        let req_ext_name = CStr::from_ptr(req_ext);

                        let found = avail_exts
                            .iter()
                            .map(|ext| ext.extension_name_as_c_str().unwrap())
                            .any(|ext_name| ext_name == req_ext_name);

                        if !found {
                            return false;
                        }
                    }

                    true
                })
                .eq(&Ok(true))
        };

        let elig_devs = phys_devs
            .iter()
            .filter_map(has_gfx_queue)
            .filter(supports_surface)
            .filter(supports_extensions)
            .map(|(phys_dev, gfx_queue_family)| {
                let props = unsafe { instance.get_physical_device_properties(phys_dev) };

                EligibleDevice {
                    phys_dev,
                    gfx_queue_family,
                    props
                }
            })
            .collect::<Vec<_>>();

        // Find an integrated GPU, if none available choose the first one
        let chosen_dev = elig_devs
            .iter()
            .find(|dev| dev.props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU)
            .unwrap_or(&elig_devs[0]);

        println!("Using device: {:?}", chosen_dev.props.device_name_as_c_str().unwrap());

        // Create logical device
        let queue_create_infos = [
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(chosen_dev.gfx_queue_family)
                .queue_priorities(&[1.0])
        ];

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&req_device_exts);

        let device = unsafe {
            instance
                .create_device(chosen_dev.phys_dev, &create_info, None)
                .unwrap()
        };

        let gfx_queue = unsafe { device.get_device_queue(chosen_dev.gfx_queue_family, 0) };

        let swapchain_ext = khr::swapchain::Device::new(&instance, &device);

        // Get surface capabilites
        let surface_capab = unsafe {
            surface_ext
                .get_physical_device_surface_capabilities(chosen_dev.phys_dev, surface)
                .unwrap()
        };

        // Calculate swapchain image extent
        let swapchain_image_extent = if surface_capab.current_extent.width != u32::MAX {
            surface_capab.current_extent
        }
        else {
            let size = window.inner_size();
    
            vk::Extent2D {
                width: cmp::max(
                    surface_capab.min_image_extent.width,
                    cmp::min(surface_capab.max_image_extent.width, size.width)
                ),
                height: cmp::max(
                    surface_capab.min_image_extent.height,
                    cmp::min(surface_capab.max_image_extent.height, size.height)
                ),
            }
        };

        // Create swapchain
        let surface_format = vk::Format::B8G8R8A8_UNORM;

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(surface_capab.min_image_count)
            .image_format(surface_format)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(swapchain_image_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capab.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);

        let swapchain = unsafe { swapchain_ext.create_swapchain(&create_info, None).unwrap() };

        // Get swapchain images
        let swapchain_images = unsafe { swapchain_ext.get_swapchain_images(swapchain).unwrap() };

        // Create swapchain image views
        let swapchain_image_views = swapchain_images
            .iter()
            .map(|&image| unsafe {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1
                    });
    
                device.create_image_view(&create_info, None).unwrap()
            })
            .collect::<Vec<_>>();

        let frames_in_flight = swapchain_images.len();
        println!("Frames in flight: {frames_in_flight}");

        // Create sync sets
        let sync_sets = (0..frames_in_flight)
            .map(|_| {
                let fence_create_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                let semaphore_create_info = vk::SemaphoreCreateInfo::default();

                unsafe {
                    let swap_image_avail = device.create_semaphore(&semaphore_create_info, None).unwrap();
                    let cmd_buf_done = device.create_semaphore(&semaphore_create_info, None).unwrap();
                    let frame_done = device.create_fence(&fence_create_info, None).unwrap();
    
                    SyncSet {
                        swap_image_avail,
                        cmd_buf_done,
                        frame_done
                    }
                }
            })
            .collect::<Vec<_>>();

        // Find needed memory types
        // We look for "direct write" memory (DEVICE_LOCAL and HOST_VISIBLE)
        // If it doesn't exist, we need to do a double copy, first to HOST_VISIBLE
        // memory, then to DEVICE_LOCAL
        let mem_props = unsafe { instance.get_physical_device_memory_properties(chosen_dev.phys_dev) };
        
        let direct_mem_type_idx = mem_props
            .memory_types_as_slice()
            .iter()
            .enumerate()
            .find_map(|(i, mem_type)| {
                mem_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE)
                    .then_some(i)
            });

        let box_bufs = match direct_mem_type_idx {
            Some(mem_type_idx) => {
                (0..frames_in_flight)
                    .map(|_| {
                        // Create buffer
                        let create_info = vk::BufferCreateInfo::default()
                            .size(BOX_BUFFER_SIZE as vk::DeviceSize)
                            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE);

                        let buf = unsafe { device.create_buffer(&create_info, None).unwrap() };

                        // Get buffer requirements
                        let reqs = unsafe { device.get_buffer_memory_requirements(buf) };

                        // Allocate memory
                        let alloc_info = vk::MemoryAllocateInfo::default()
                            .allocation_size(reqs.size)
                            .memory_type_index(mem_type_idx as u32);

                        let mem = unsafe { device.allocate_memory(&alloc_info, None).unwrap() };

                        // Map memory
                        let ptr = unsafe {
                            device
                                .map_memory(mem, 0, BOX_BUFFER_SIZE as vk::DeviceSize, vk::MemoryMapFlags::empty())
                                .unwrap()
                        };

                        // Bind memory to buffer
                        unsafe { device.bind_buffer_memory(buf, mem, 0).unwrap(); }

                        BoxBuffer {
                            buf: (buf, mem, ptr as *mut u8),
                            staging_buf: None,
                        }
                    })
                    .collect::<Vec<_>>()
            }

            None => todo!()
        };

        // Create command pool and buffers
        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(chosen_dev.gfx_queue_family);            

        let cmd_pool = unsafe { device.create_command_pool(&create_info, None).unwrap() };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(frames_in_flight as u32);

        let cmd_bufs = unsafe { device.allocate_command_buffers(&alloc_info).unwrap() };

        // Create descriptor pool
        // There will be frames_in_flight number of sets, each set will have
        // one storage image descriptor (for the swapchain output image) and one
        // storage buffer descriptor (for the box buffer)
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(frames_in_flight as u32),

            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(frames_in_flight as u32),
        ];

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(frames_in_flight as u32)
            .pool_sizes(&pool_sizes);

        let desc_pool = unsafe { device.create_descriptor_pool(&create_info, None).unwrap() };

        // Create descriptor set layout
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),

            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_set_layout = unsafe { device.create_descriptor_set_layout(&create_info, None).unwrap() };

        // Create descriptor sets
        let set_layouts = vec![desc_set_layout; frames_in_flight];

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);

        let desc_sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

        // Write descriptors
        let image_infos = swapchain_image_views
            .iter()
            .map(|view| {
                let info = vk::DescriptorImageInfo::default()
                    .sampler(vk::Sampler::null())
                    .image_view(*view)
                    .image_layout(vk::ImageLayout::GENERAL);

                [info]
            })
            .collect::<Vec<_>>();

        let buf_infos = box_bufs
            .iter()
            .map(|buf| {
                let info = vk::DescriptorBufferInfo::default()
                    .buffer(buf.buf.0)
                    .offset(0)
                    .range(BOX_BUFFER_SIZE as vk::DeviceSize);

                [info]
            })
            .collect::<Vec<_>>();

        let image_writes = desc_sets
            .iter()
            .zip(&image_infos)
            .map(|(desc_set, image_info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(*desc_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(image_info)
            });

        let buf_writes = desc_sets
            .iter()
            .zip(&buf_infos)
            .map(|(desc_set, buf_info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(*desc_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(buf_info)
            });

        let writes = image_writes.chain(buf_writes).collect::<Vec<_>>();

        unsafe { device.update_descriptor_sets(&writes, &[]); }

        // Create shader module
        let create_info = vk::ShaderModuleCreateInfo {
            code_size: SHADER_CODE.len(),
            p_code: SHADER_CODE.as_ptr() as *const u32,
            ..Default::default()
        };

        let shader_module = unsafe { device.create_shader_module(&create_info, None).unwrap() };

        // Create pipeline layout
        let desc_set_layouts = [desc_set_layout];
        let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&desc_set_layouts);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&create_info, None).unwrap() };

        // Create pipeline
        let shader_main = CString::new("main").unwrap();

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&shader_main);

        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .unwrap()[0]
        };

        Self {
            frame_idx: 0,
            pipeline_layout,
            pipeline,
            desc_pool,
            desc_sets,
            cmd_bufs,
            cmd_pool,
            box_bufs,
            sync_sets,
            swapchain_image_extent,
            swapchain_images,
            swapchain_ext,
            swapchain,
            device,
            gfx_queue,
            surface,
            instance,
            _entry: entry
        }
    }

    pub fn run(&mut self, dom: &Mutex<Dom>, should_exit: impl Fn() -> bool) {
        loop {
            if should_exit() {
                break;
            }

            let sync_set = &self.sync_sets[self.frame_idx];
            let cmd_buf = self.cmd_bufs[self.frame_idx];
            let desc_set = self.desc_sets[self.frame_idx];
            let box_buf = &self.box_bufs[self.frame_idx];
            let device = &self.device;

            let frames_in_flight = self.sync_sets.len();
            self.frame_idx = (self.frame_idx + 1) % frames_in_flight;

            unsafe {
                // Wait for previous frame in this slot to finish
                device.wait_for_fences(&[sync_set.frame_done], true, u64::MAX).unwrap();
                device.reset_fences(&[sync_set.frame_done]).unwrap();

                // Acquire swapchain image
                let (swapchain_image_idx, _is_suboptimal) = self.swapchain_ext
                    .acquire_next_image(self.swapchain, u64::MAX, sync_set.swap_image_avail, vk::Fence::null())
                    .unwrap();

                let swapchain_image = self.swapchain_images[swapchain_image_idx as usize];

                // Begin command buffer recording
                device.begin_command_buffer(cmd_buf, &vk::CommandBufferBeginInfo::default()).unwrap();

                // Transition swapchain image layout from UNDEFINED to GENERAL
                let barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(swapchain_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier]
                );

                // Draw the DOM
                device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, self.pipeline);
                device.cmd_bind_descriptor_sets(cmd_buf, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[desc_set], &[]);

                let box_buf = &mut *(box_buf.buf.2 as *mut BoxBufferContents);

                // Lock the DOM for as little time as possible
                {
                    let mut dom = dom.lock().unwrap();

                    let root_node = dom.root_node.clone();

                    // Compute layout
                    let avail_space = Size {
                        height: AvailableSpace::Definite(self.swapchain_image_extent.height as f32),
                        width: AvailableSpace::Definite(self.swapchain_image_extent.width as f32)
                    };

                    dom.layout_tree.compute_layout(root_node, avail_space);
                    
                    // Write box info to the box buffer
                    assert!(dom.box_list.len() <= BOX_LIST_LEN);

                    box_buf.len = dom.box_list.len() as u32;

                    for (i, (key, dom_box)) in dom.box_list.iter().enumerate() {
                        let layout = dom.layout_tree.layout(key.into());

                        let pos = layout.location;
                        let size = layout.size;

                        box_buf.boxes[i] = Box {
                            pos: [pos.x, pos.y],
                            size: [size.width, size.height],
                            color: dom_box.background_color.as_f32_array(),
                            radius: dom_box.radius,
                            padding: [0.0; 3]
                        };
                    }
                }

                let workgroups_x = (self.swapchain_image_extent.width + WG_SIZE - 1) / WG_SIZE;
                let workgroups_y = (self.swapchain_image_extent.height + WG_SIZE - 1) / WG_SIZE;

                device.cmd_dispatch(cmd_buf, workgroups_x, workgroups_y, 1);

                // Transition swapchain image layout from GENERAL TO PRESENT_SRC_KHR
                let barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(swapchain_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                
                self.device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier]
                );

                // End command buffer recording
                device.end_command_buffer(cmd_buf).unwrap();

                // Submit command buffer
                let wait_semaphores = [sync_set.swap_image_avail];
                let wait_stages = [vk::PipelineStageFlags::ALL_COMMANDS];
                
                let cmd_bufs = [cmd_buf];
                let signal_semaphores = [sync_set.cmd_buf_done];
                
                let submit_info = vk::SubmitInfo::default()
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&wait_stages)
                    .command_buffers(&cmd_bufs)
                    .signal_semaphores(&signal_semaphores);
                    
                device.queue_submit(self.gfx_queue, &[submit_info], sync_set.frame_done).unwrap();

                // Present frame        
                let wait_semaphores = [sync_set.cmd_buf_done];
                let swapchains = [self.swapchain];
                let image_indices = [swapchain_image_idx];
                
                let present_info = vk::PresentInfoKHR::default()
                    .wait_semaphores(&wait_semaphores)
                    .swapchains(&swapchains)
                    .image_indices(&image_indices);
                
                self.swapchain_ext.queue_present(self.gfx_queue, &present_info).unwrap();
            }
        }
    }
}
