
pub extern crate cgmath;
pub extern crate gfx_hal;
pub extern crate gfx_mem;
pub extern crate xfg;

#[macro_use] extern crate log;
extern crate env_logger;
extern crate winit;

use std::sync::Arc;
use std::time::{Duration, Instant};

use cgmath::{Deg, PerspectiveFov, Matrix4, SquareMatrix};
use gfx_hal::{Backend, Device, Instance, PhysicalDevice, Surface};
use gfx_hal::command::{Rect, Viewport};
use gfx_hal::device::{Extent, WaitFor};
use gfx_hal::format::{ChannelType, Format};
use gfx_hal::memory::{Properties};
use gfx_hal::pool::{CommandPool, CommandPoolCreateFlags};
use gfx_hal::queue::Graphics;
use gfx_hal::window::{FrameSync, Swapchain, SwapchainConfig};

use gfx_mem::{Factory, SmartAllocator, Type};

use winit::{EventsLoop, WindowBuilder};

use xfg::{Pass, SuperFrame, GraphBuilder};

#[cfg(feature = "dx12")]
pub extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
pub extern crate gfx_backend_metal as back;
#[cfg(feature = "gl")]
pub extern crate gfx_backend_gl as back;
#[cfg(feature = "vulkan")]
pub extern crate gfx_backend_vulkan as back;
#[cfg(not(any(feature = "dx12", feature = "metal", feature = "gl", feature = "vulkan")))]
pub extern crate gfx_backend_empty as back;


pub type Buffer<B> = <SmartAllocator<B> as Factory<B>>::Buffer;
pub type Image<B> = <SmartAllocator<B> as Factory<B>>::Image;
pub const REQUEST_DEVICE_LOCAL: (Type, Properties) = (Type::General, Properties::DEVICE_LOCAL);
pub const REQUEST_CPU_VISIBLE: (Type, Properties) = (Type::General, Properties::CPU_VISIBLE);

pub struct Cache<B: Backend> {
    pub uniforms: Vec<Buffer<B>>,
    pub views: Vec<B::ImageView>,
    pub set: B::DescriptorSet,
}

pub struct Mesh<B: Backend> {
    pub indices: Buffer<B>,
    pub vertices: Buffer<B>,
    pub index_count: u32,
}

pub struct Object<B: Backend, T = ()> {
    pub mesh: Arc<Mesh<B>>,
    pub transform: Matrix4<f32>,
    pub data: T,
    pub cache: Vec<Option<Cache<B>>>,
}

pub struct Light<B: Backend> {
    pub color: [f32; 3],
    pub transform: Matrix4<f32>,
    pub cache: Vec<Option<Cache<B>>>,
}

#[derive(Clone, Copy, Debug)]
pub struct AmbientLight(pub [f32; 3]);

pub struct Camera {
    pub transform: Matrix4<f32>,
    pub projection: Matrix4<f32>,
}

pub struct Scene<B: Backend, T = ()> {
    pub objects: Vec<Object<B, T>>,
    pub ambient: AmbientLight,
    pub lights: Vec<Light<B>>,
    pub camera: Camera,
    pub allocator: SmartAllocator<B>,
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "gl", feature = "vulkan")))]
pub fn run<T, Y>(_: T, _: Y) {
    env_logger::init();
    error!("You need to enable the native API feature (vulkan/metal/dx12/gl) in order to run example");
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "gl", feature = "vulkan"))]
#[deny(dead_code)]
pub fn run<G, F, P, T>(graph: G, fill: F)
where
    G: FnOnce(Format, &mut GraphBuilder<P>),
    P: Pass<back::Backend, Scene<back::Backend, T>>,
    F: FnOnce(&mut Scene<back::Backend, T>, &<back::Backend as Backend>::Device),
{
    env_logger::init();

    #[cfg(feature = "metal")]
    let mut autorelease_pool = unsafe { back::AutoreleasePool::new() };

    let mut events_loop = EventsLoop::new();

    let wb = WindowBuilder::new()
        .with_dimensions(960, 640)
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

    events_loop.poll_events(|_| ());

    let (width, height) = window.get_inner_size().unwrap();
    let hidpi = window.hidpi_factor();
    info!("Width: {}, Height: {}, HIDPI: {}", width, height, hidpi);
    
    #[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
    let (_instance, adapter, mut surface) = {
        let instance = back::Instance::create("gfx-rs quad", 1);
        let surface = instance.create_surface(&window);
        let mut adapters = instance.enumerate_adapters();
        (instance, adapters.remove(0), surface)
    };
    #[cfg(feature = "gl")]
    let (adapter, mut surface) = {
        let surface = back::Surface::from_window(window);
        let mut adapters = surface.enumerate_adapters();
        (adapters.remove(0), surface)
    };

    events_loop.poll_events(|_| ());

    let surface_format = surface
        .capabilities_and_formats(&adapter.physical_device)
        .1
        .and_then(
            |formats| {
                formats
                    .into_iter()
                    .find(|format| {
                        format.base_format().1 == ChannelType::Srgb
                    })
            }
        ).unwrap_or(Format::Rgba8Srgb);

    let memory_properties = adapter
        .physical_device
        .memory_properties();

    let mut allocator = SmartAllocator::<back::Backend>::new(memory_properties, 32, 32, 32, 1024 * 1024 * 64);

    info!("Device features: {:#?}", adapter.physical_device.get_features());
    info!("Device limits: {:#?}", adapter.physical_device.get_limits());

    let (device, mut queue_group) =
        adapter.open_with::<_, Graphics>(1, |family| {
            surface.supports_queue_family(family)
        }).unwrap();

    let buffering = 3;

    let swap_config = SwapchainConfig::new()
        .with_color(surface_format)
        .with_image_count(buffering);
    let (mut swap_chain, backbuffer) = device.create_swapchain(&mut surface, swap_config);

    let mut command_pools = (0..buffering).map(|_| device.create_command_pool_typed(&queue_group, CommandPoolCreateFlags::empty(), 16)).collect::<Vec<_>>();
    let mut command_queue = &mut queue_group.queues[0];

    events_loop.poll_events(|_| ());

    let mut graph = {
        let mut builder = GraphBuilder::new();
        graph(surface_format, &mut builder);
        builder
            .with_extent(Extent { width: width as u32, height: height as u32, depth: 1 })
            .build(&device, &backbuffer, |kind, level, format, usage, properties, device| {
                allocator.create_image(device, (Type::General, properties), kind, level, format, usage)
            }).unwrap()
    };

    let projection: Matrix4<f32> = PerspectiveFov {
        fovy: Deg(60.0).into(),
        aspect: (width as f32) / (height as f32),
        near: 0.1,
        far: 2000.0,
    }.into();

    let mut scene = Scene {
        objects: Vec::new(),
        ambient: AmbientLight([0.0, 0.0, 0.0]),
        lights: Vec::new(),
        camera: Camera {
            transform: Matrix4::identity(),
            projection,
        },
        allocator,
    };

    // fill scene
    fill(&mut scene, &device);

    let mut acquires = (0..buffering+1).map(|_| device.create_semaphore()).collect::<Vec<_>>();
    let mut releases = (0..buffering).map(|_| device.create_semaphore()).collect::<Vec<_>>();
    let mut finishes = (0..buffering).map(|_| device.create_fence(false)).collect::<Vec<_>>();

    struct Job<B: Backend> {
        acquire: B::Semaphore,
        release: B::Semaphore,
        finish: B::Fence,
        command_pool: CommandPool<B, Graphics>,
    }

    let mut jobs: Vec<Option<Job<_>>> = (0..buffering).map(|_| None).collect();

    let start = Instant::now();
    let mut total = 0;
    let total = loop {
        total += 1;
        // info!("Iteration: {}", i);
        // info!("Poll events");
        events_loop.poll_events(|_| ());

        // There is always one unused.
        let acquire = acquires.pop().unwrap();

        // info!("Acquire frame");
        let frame = swap_chain.acquire_frame(FrameSync::Semaphore(&acquire));
        let id = frame.id();

        if let Some(mut job) = jobs[id].take() {
            if !device.wait_for_fences(Some(&job.finish), WaitFor::All, !0) {
                panic!("Failed to wait for drawing");
            }
            device.reset_fences(Some(&job.finish));
            job.command_pool.reset();

            #[cfg(feature = "metal")]
            unsafe {
                autorelease_pool.reset();
            }

            acquires.push(job.acquire);
            releases.push(job.release);
            finishes.push(job.finish);
            command_pools.push(job.command_pool);
        }

        let release = releases.pop().unwrap();
        let finish = finishes.pop().unwrap();
        let mut command_pool = command_pools.pop().unwrap();

        let frame = SuperFrame::new(&backbuffer, frame);

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
                    w: width as u16,
                    h: height as u16,
                },
                depth: 0.0 .. 1.0,
            },
            &finish,
            &device,
            &mut scene,
        );

        swap_chain.present(&mut command_queue, Some(&release));

        jobs[id] = Some(Job {
            acquire,
            release,
            finish,
            command_pool,
        });

        if Instant::now() - start > Duration::from_secs(10) {
            break total;
        }
    };
    
    if !device.wait_for_fences(jobs.iter().filter_map(|job| job.as_ref().map(|job| &job.finish)), WaitFor::All, !0) {
        panic!("Failed to wait for drawing");
    }
    device.reset_fences(jobs.iter().filter_map(|job| job.as_ref().map(|job| &job.finish)));
    
    for id in 0 .. jobs.len() {
        if let Some(mut job) = jobs[id].take() {
            job.command_pool.reset();

            #[cfg(feature = "metal")]
            unsafe {
                autorelease_pool.reset();
            }

            acquires.push(job.acquire);
            releases.push(job.release);
            finishes.push(job.finish);
            command_pools.push(job.command_pool);
        }
    }

    let end = Instant::now();
    let dur = end - start;
    let fps = (total as f64) / (dur.as_secs() as f64 + dur.subsec_nanos() as f64 / 1000000000f64);
    info!("Run time: {}.{:09}", dur.as_secs(), dur.subsec_nanos());
    info!("Total frames rendered: {}", total);
    info!("Average FPS: {}", fps);

    // TODO: Dispose everything properly.
    ::std::process::exit(0);
}
