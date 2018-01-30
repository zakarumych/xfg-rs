
extern crate cgmath;
extern crate env_logger;
extern crate gfx_hal;
extern crate gfx_mem;
extern crate smallvec;
extern crate winit;
extern crate xfg;

use std::borrow::Borrow;

use cgmath::{Deg, Matrix4, PerspectiveFov, SquareMatrix};
use gfx_hal::{Backend, Device, IndexType};
use gfx_hal::buffer::{IndexBufferView, Usage};
use gfx_hal::command::{CommandBuffer, RenderPassInlineEncoder, Primary};
use gfx_hal::device::ShaderError;
use gfx_hal::format::Format;
use gfx_hal::memory::{cast_slice, Pod, Properties};
use gfx_hal::pso::{DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType, DescriptorWrite, Element, ElemStride, EntryPoint, GraphicsShaderSet, ShaderStageFlags, VertexBufferSet};
use gfx_hal::queue::Transfer;
use gfx_mem::{Block, Factory, SmartAllocator, Type};
use smallvec::SmallVec;
use xfg::{DescriptorPool, Pass};

type Buffer<B> = <SmartAllocator<B> as Factory<B>>::Buffer;
const REQUEST_DEVICE_LOCAL: (Type, Properties) = (Type::General, Properties::DEVICE_LOCAL);
const REQUEST_CPU_VISIBLE: (Type, Properties) = (Type::General, Properties::CPU_VISIBLE);

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct TrProjView {
    transform: Matrix4<f32>,
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

unsafe impl Pod for TrProjView {}

struct Cache<B: Backend> {
    uniform: Buffer<B>,
    set: B::DescriptorSet,
}

struct Object<B: Backend> {
    indices: Buffer<B>,
    pos_color: Buffer<B>,
    index_count: u32,
    transform: Matrix4<f32>,
    cache: Option<Cache<B>>,
}

struct Camera {
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

struct Scene<B: Backend> {
    objects: Vec<Object<B>>,
    camera: Camera,
    allocator: SmartAllocator<B>,
}

#[derive(Debug)]
struct DrawFlat;

impl<B> Pass<B, Scene<B>> for DrawFlat
where
    B: Backend,
{
    /// Name of the pass
    fn name(&self) -> &str {
        "DrawFlat"
    }

    /// Input attachments
    fn inputs(&self) -> usize { 0 }

    /// Color attachments
    fn colors(&self) -> usize { 1 }

    /// Uses depth attachment
    fn depth(&self) -> bool { true }

    /// Uses stencil attachment
    fn stencil(&self) -> bool { false }

    /// Vertices format
    fn vertices(&self) -> &[(&[Element<Format>], ElemStride)] {
        &[
            (
                &[
                    Element {
                        format: Format::Rgb32Float,
                        offset: 0,
                    },
                    Element {
                        format: Format::Rgba32Float,
                        offset: 12,
                    },
                ],
                28,
            )
        ]
    }

    fn bindings(&self) -> &[DescriptorSetLayoutBinding] {
        &[
            DescriptorSetLayoutBinding {
                binding: 0,
                ty: DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: ShaderStageFlags::VERTEX,
            }
        ]
    }

    fn shaders<'a>(
        &self,
        shaders: &'a mut SmallVec<[B::ShaderModule; 5]>,
        device: &B::Device,
    ) -> Result<GraphicsShaderSet<'a, B>, ShaderError> {
        shaders.clear();
        shaders.push(device.create_shader_module(include_bytes!("vert.spv"))?);
        shaders.push(device.create_shader_module(include_bytes!("frag.spv"))?);

        Ok(GraphicsShaderSet {
            vertex: EntryPoint {
                entry: "main",
                module: &shaders[0],
                specialization: &[],
            },
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(EntryPoint {
                entry: "main",
                module: &shaders[1],
                specialization: &[],
            }),
        })
    }

    fn prepare<'a>(
        &mut self,
        pool: &mut DescriptorPool<B>,
        cbuf: &mut CommandBuffer<B, Transfer>,
        device: &B::Device,
        scene: &mut Scene<B>,
    )
    {
        let ref mut allocator = scene.allocator;
        // Update uniform cache
        for obj in &mut scene.objects {
            let trprojview = TrProjView {
                transform: obj.transform,
                projection: scene.camera.proj,
                view: scene.camera.view,
            };

            let cache = obj.cache.get_or_insert_with(|| {
                let buffer = allocator.create_buffer(device, REQUEST_DEVICE_LOCAL, ::std::mem::size_of::<TrProjView>() as u64, Usage::UNIFORM).unwrap();
                let set = pool.allocate(device);
                device.update_descriptor_sets(&[DescriptorSetWrite {
                    set: &set,
                    binding: 0,
                    array_offset: 0,
                    write: DescriptorWrite::UniformBuffer(vec![(buffer.borrow(), buffer.range())]),
                }]);
                Cache {
                    uniform: buffer,
                    set,
                }
            });
            cbuf.update_buffer(cache.uniform.borrow(), 0, cast_slice(&[trprojview]));
        }
    }

    fn draw_inline<'a>(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassInlineEncoder<B, Primary>,
        _device: &B::Device,
        scene: &Scene<B>,
    ) {       
        for object in &scene.objects {
            let cache = object.cache.as_ref().unwrap();
            encoder.bind_graphics_descriptor_sets(layout, 0, Some(&cache.set));
            encoder.bind_index_buffer(IndexBufferView {
                buffer: object.indices.borrow(),
                offset: 0,
                index_type: IndexType::U16,
            });
            encoder.bind_vertex_buffers(VertexBufferSet(vec![(object.pos_color.borrow(), 0)]));
            encoder.draw_indexed(
                0 .. object.index_count,
                0,
                0 .. 1,
            );
        }
    }

    fn cleanup(&mut self, pool: &mut DescriptorPool<B>, device: &B::Device, scene: &mut Scene<B>) {
        for object in &mut scene.objects {
            if let Some(cache) = object.cache.take() {
                pool.free(cache.set);
                scene.allocator.destroy_buffer(device, cache.uniform);
            }
        }
    }
}

fn create_cube<B>(device: &B::Device, factory: &mut SmartAllocator<B>, transform: Matrix4<f32>) -> Object<B>
where
    B: Backend,
{
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct PosColor {
        position: [f32; 3],
        color: [f32; 4],
    }

    unsafe impl Pod for PosColor {}

    let pos_color = vec![
        // Right
        PosColor {
            position: [0.5, -0.5, -0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, -0.5, 0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, 0.5, -0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, 0.5, 0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        // Left
        PosColor {
            position: [-0.5, -0.5, -0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PosColor {
            position: [-0.5, -0.5, 0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PosColor {
            position: [-0.5, 0.5, -0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PosColor {
            position: [-0.5, 0.5, 0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        // Top
        PosColor {
            position: [-0.5, 0.5, -0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        PosColor {
            position: [-0.5, 0.5, 0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, 0.5, -0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, 0.5, 0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        // Bottom
        PosColor {
            position: [-0.5, -0.5, -0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        PosColor {
            position: [-0.5, -0.5, 0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, -0.5, -0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, -0.5, 0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        // Front
        PosColor {
            position: [-0.5, -0.5, 0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        PosColor {
            position: [-0.5, 0.5, 0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        PosColor {
            position: [0.5, -0.5, 0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        PosColor {
            position: [0.5, 0.5, 0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        // Back
        PosColor {
            position: [-0.5, -0.5, -0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        PosColor {
            position: [-0.5, 0.5, -0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        PosColor {
            position: [0.5, -0.5, -0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        PosColor {
            position: [0.5, 0.5, -0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
    ];

    let pos_color: &[u8] = cast_slice(&pos_color);

    let buffer = factory.create_buffer(device, REQUEST_CPU_VISIBLE, pos_color.len() as u64, Usage::VERTEX).unwrap();
    {
        let start = buffer.range().start;
        let end = start + pos_color.len() as u64;
        let mut writer = device.acquire_mapping_writer(buffer.memory(), start .. end).unwrap();
        writer.copy_from_slice(pos_color);
        device.release_mapping_writer(writer);
    }

    let pos_color = buffer;

    let indices: Vec<u16> = vec![
        // Left
        0, 1, 2,
        1, 2, 3,
        // Right
        4, 5, 6,
        5, 6, 7,
        // Top
        8, 9, 10,
        9, 10, 11,
        // Bottom
        12, 13, 14,
        13, 14, 15,
        // Front
        16, 17, 18,
        17, 18, 19,
        // Back
        20, 21, 22,
        21, 22, 23,
    ];

    let index_count = indices.len() as u32;

    let indices: &[u8] = cast_slice(&indices);

    let buffer = factory.create_buffer(device, REQUEST_CPU_VISIBLE, indices.len() as u64, Usage::INDEX).unwrap();
    {
        let mut writer = device.acquire_mapping_writer(buffer.memory(), buffer.range()).unwrap();
        writer.copy_from_slice(indices);
        device.release_mapping_writer(writer);
    }

    let indices = buffer;

    Object {
        pos_color,
        indices,
        index_count,
        transform,
        cache: None,
    }
}

fn main() {
    #[cfg(feature = "dx12")]
    extern crate gfx_backend_dx12 as back;
    #[cfg(feature = "metal")]
    extern crate gfx_backend_metal as back;
    #[cfg(feature = "gl")]
    extern crate gfx_backend_gl as back;
    #[cfg(feature = "vulkan")]
    extern crate gfx_backend_vulkan as back;

    use gfx_hal::{Instance, PhysicalDevice, Surface};
    use gfx_hal::command::{ClearColor, ClearDepthStencil, Rect, Viewport};
    use gfx_hal::device::{Extent, WaitFor};
    use gfx_hal::format::ChannelType;
    use gfx_hal::pool::CommandPoolCreateFlags;
    use gfx_hal::queue::Graphics;
    use gfx_hal::window::{FrameSync, Swapchain, SwapchainConfig};

    use winit::{EventsLoop, WindowBuilder};

    use xfg::{ColorAttachment, DepthStencilAttachment, SuperFrame, GraphBuilder};

    env_logger::init();

    let mut events_loop = EventsLoop::new();

    let wb = WindowBuilder::new()
        .with_dimensions(480, 480)
        .with_title("flat".to_string());

    #[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
    let window = wb
        .build(&events_loop)
        .unwrap();
    #[cfg(feature = "gl")]
    let window = {
        let builder = back::config_context(
            back::glutin::ContextBuilder::new(),
            Format::Rgba8Srgb,
            None,
        ).with_vsync(true);
        back::glutin::GlWindow::new(wb, builder, &events_loop).unwrap()
    };
    
    #[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
    let (_instance, mut adapter, mut surface) = {
        let instance = back::Instance::create("gfx-rs quad", 1);
        let surface = instance.create_surface(&window);
        let mut adapters = instance.enumerate_adapters();
        (instance, adapters.remove(0), surface)
    };
    #[cfg(feature = "gl")]
    let (mut adapter, mut surface) = {
        let surface = back::Surface::from_window(window);
        let mut adapters = surface.enumerate_adapters();
        (adapters.remove(0), surface)
    };

    let surface_format = surface
        .capabilities_and_formats(&adapter.physical_device)
        .1
        .map_or(
            Format::Rgba8Srgb,
            |formats| {
                formats
                    .into_iter()
                    .find(|format| {
                        format.base_format().1 == ChannelType::Srgb
                    })
                    .unwrap()
            }
        );

    let memory_properties = adapter
        .physical_device
        .memory_properties();

    let mut allocator = SmartAllocator::<back::Backend>::new(memory_properties, 32, 32, 32, 480 * 480 * 64);

    let (device, mut queue_group) =
        adapter.open_with::<_, Graphics>(1, |family| {
            surface.supports_queue_family(family)
        }).unwrap();

    let mut command_pool = device.create_command_pool_typed(&queue_group, CommandPoolCreateFlags::empty(), 16);
    let mut command_queue = &mut queue_group.queues[0];

    let swap_config = SwapchainConfig::new()
        .with_color(surface_format)
        .with_image_count(1);
    let (mut swap_chain, backbuffer) = device.create_swapchain(&mut surface, swap_config);

    let mut graph = {
        let depth = DepthStencilAttachment::new(Format::D32Float).with_clear(ClearDepthStencil(1.0, 0));
        let present = ColorAttachment::new(surface_format).with_clear(ClearColor::Float([0.3, 0.4, 0.5, 1.0]));
        let pass = DrawFlat.build()
            .with_color(0, &present)
            .with_depth(&depth)
            ;
        GraphBuilder::new()
            .with_pass(pass)
            .with_extent(Extent { width: 480, height: 480, depth: 1 })
            .with_backbuffer(&backbuffer)
            .with_present(&present)
            .build(&device, |kind, level, format, usage, properties, device| {
                allocator.create_image(device, (Type::General, properties), kind, level, format, usage)
            }).unwrap()
    };

    let view = Matrix4::from_translation([0.0, 0.0, -5.0].into());
    let view = view * Matrix4::from_angle_x(Deg(-45.0));
    let view = view * Matrix4::from_angle_y(Deg(25.0));

    let proj: Matrix4<f32> = PerspectiveFov {
        fovy: Deg(60.0).into(),
        aspect: 1.0,
        near: 0.1,
        far: 2000.0,
    }.into();

    let mut scene = Scene {
        objects: vec![], //vec![create_cube(&device, &mut allocator, Matrix4::identity())],
        camera: Camera {
            view,
            proj,
        },
        allocator,
    };

    let acquire = device.create_semaphore();
    let release = device.create_semaphore();
    let finish = device.create_fence(false);

    for i in 0 .. 10 {
        println!("Iteration: {}", i);
        println!("Poll events");
        events_loop.poll_events(|_| ());

        println!("Sleep a bit");
        ::std::thread::sleep(::std::time::Duration::from_millis(1000));

        println!("Acquire frame");
        let frame = SuperFrame::new(&backbuffer, swap_chain.acquire_frame(FrameSync::Semaphore(&acquire)));

        println!("Draw inline");
        graph.draw_inline(
            &mut command_queue,
            &mut command_pool,
            frame,
            &acquire,
            &release,
            Viewport {
                rect: Rect {
                    x: 0,
                    y: 0,
                    w: 480,
                    h: 480,
                },
                depth: 0.0 .. 1.0,
            },
            &finish,
            &device,
            &mut scene,
        );

        println!("Present frame");
        swap_chain.present(&mut command_queue, Some(&release));

        println!("Wait for idle");
        if !device.wait_for_fences(&[&finish], WaitFor::All, 1000) {
            panic!("Failed to wait for drawing in 1 sec");
        }

        device.reset_fences(&[&finish]);

        println!("Reset pool");
        command_pool.reset();
    }

    println!("FINISH");
    ::std::process::exit(0);
}