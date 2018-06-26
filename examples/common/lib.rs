
extern crate cgmath;
extern crate gfx_hal as hal;
extern crate gfx_mesh as mesh;
extern crate gfx_render as gfx;
extern crate gfx_memory as memory;
extern crate genmesh;
extern crate smallvec;
extern crate xfg;

#[macro_use]
extern crate glsl_layout;

extern crate env_logger;
#[macro_use]
extern crate log;
extern crate winit;

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as backend;
#[cfg(not(any(feature = "vulkan", feature = "dx12", feature = "metal")))]
extern crate gfx_backend_empty as backend;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as backend;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as backend;

#[cfg(feature = "profile")]
extern crate flame;

#[cfg(feature = "profile")]
macro_rules! profile {
    ($name:tt) => {
        let _guard = ::flame::start_guard(concat!("'", $name, "' @ ", concat!(file!(), ":", line!())));
    }
}

#[cfg(not(feature = "profile"))]
macro_rules! profile {
    ($name:tt) => {}
}

type Back = backend::Backend;

use std::{
    borrow::{Borrow, Cow}, cell::UnsafeCell, collections::HashMap, sync::Arc,
    iter::{once, empty},
    time::{Duration, Instant},
    ops::{Index, Range},
    mem::{size_of, zeroed},
};

use cgmath::{Deg, Point3, Matrix4, PerspectiveFov, SquareMatrix, Transform, EuclideanSpace};

use gfx::{BackendEx, Buffer, Factory, Image, Render, Renderer};
use memory::Block;

use glsl_layout::*;

use hal::{
    buffer, command::{
        ClearValue, ClearColor, ClearDepthStencil, CommandBuffer, Primary, RenderPassInlineEncoder,
        DescriptorSetOffset, BufferImageCopy,
    },
    device::WaitFor,
    format::{AsFormat, ChannelType, Format, Swizzle, Aspects}, image, image::{Extent, StorageFlags, Tiling},
    memory::{cast_slice, Barrier, Dependencies, Pod, Properties},
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{
        AllocationError, Descriptor, DescriptorPool, DescriptorRangeDesc,
        DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType, ElemStride, Element,
        EntryPoint, GraphicsShaderSet, PipelineStage, Rect, ShaderStageFlags, VertexBufferSet,
        Viewport, ColorBlendDesc, ColorMask, BlendState, DepthTest, Comparison, StencilTest, DepthStencilDesc,
    },
    queue::{Graphics, QueueFamilyId},
    window::{Backbuffer, Extent2D, FrameSync, Swapchain, SwapchainConfig}, Backend, Device,
    Instance, PhysicalDevice, Surface,
};

use mesh::*;

use smallvec::SmallVec;

use winit::{EventsLoop, WindowBuilder};

use xfg::render::*;
use xfg::*;

/// Descriptor pool that caches set allocations.
#[derive(Debug)]
pub struct XfgDescriptorPool<B: Backend> {
    pools: Vec<B::DescriptorPool>,
    sets: Vec<B::DescriptorSet>,
}

impl<B> XfgDescriptorPool<B>
where
    B: Backend,
{
    /// Create empty pool.
    pub fn new() -> Self {
        XfgDescriptorPool {
            pools: Vec::new(),
            sets: Vec::new(),
        }
    }

    /// Allocate descriptor set for the render pass.
    pub fn allocate<D>(
        &mut self,
        device: &mut D,
        layout: &B::DescriptorSetLayout,
        bindings: &[DescriptorSetLayoutBinding]
    ) -> B::DescriptorSet
    where
        D: Device<B>,
    {
        let ref mut pools = self.pools;
        let exp = pools.len() as u32;

        self.sets.pop().unwrap_or_else(|| {
            pools
                .last_mut()
                .and_then(|pool| match pool.allocate_set(layout) {
                    Ok(set) => Some(set),
                    Err(AllocationError::OutOfPoolMemory) | Err(AllocationError::FragmentedPool) => None,
                    Err(AllocationError::IncompatibleLayout) => unreachable!(),
                    Err(err) => panic!("Unhandled error in XfgDescriptorPool: {}", err),
                })
                .unwrap_or_else(|| {
                    let multiplier = 2usize.pow(exp);
                    let mut pool = device.create_descriptor_pool(
                        multiplier,
                        bindings_to_range(multiplier, bindings),
                    );

                    let set = match pool.allocate_set(layout) {
                        Ok(set) => set,
                        Err(AllocationError::OutOfPoolMemory)
                        | Err(AllocationError::FragmentedPool)
                        | Err(AllocationError::IncompatibleLayout) => unreachable!(),
                        Err(err) => panic!("Unhandled error in XfgDescriptorPool: {}", err),
                    };

                    pools.push(pool);
                    set
                })
        })
    }

    /// Free descriptor set.
    pub fn free(&mut self, set: B::DescriptorSet) {
        self.sets.push(set)
    }
}

pub struct Cache<B: Backend> {
    pub uniforms: Vec<Buffer<B>>,
    pub set: B::DescriptorSet,
}

pub struct Object<B: Backend, T = ()> {
    pub mesh: Arc<Mesh<B>>,
    pub transform: Matrix4<f32>,
    pub data: T,
    pub cache: UnsafeCell<Option<Cache<B>>>,
}

pub struct Light<B: Backend> {
    pub color: [f32; 3],
    pub transform: Matrix4<f32>,
    pub cache: UnsafeCell<Option<Cache<B>>>,
}

#[derive(Clone, Copy, Debug)]
pub struct AmbientLight(pub [f32; 3]);

pub struct Camera {
    pub transform: Matrix4<f32>,
    pub projection: Matrix4<f32>,
}

pub struct Scene<B: Backend, T = (), Y = ()> {
    pub camera: Camera,
    pub ambient: AmbientLight,
    pub lights: Vec<Light<B>>,
    pub objects: Vec<Object<B, T>>,
    pub other: Option<Y>,
}

type XfgGraphBuilder<B, T = (), Y = ()> = GraphBuilder<B, Factory<B>, Scene<B, T, Y>, Buffer<B>, Image<B>>;

struct XfgGraph<B: Backend, T = (), Y = ()>(Graph<B, Factory<B>, Scene<B, T, Y>, Buffer<B>, Image<B>>);

impl<B, T, Y> From<Graph<B, Factory<B>, Scene<B, T, Y>, Buffer<B>, Image<B>>> for XfgGraph<B, T, Y>
where
    B: Backend,
{
    fn from(graph: Graph<B, Factory<B>, Scene<B, T, Y>, Buffer<B>, Image<B>>) -> Self {
        XfgGraph(graph)
    }
}

impl<B, T, Y> Render<B, Scene<B, T, Y>> for XfgGraph<B, T, Y>
where
    B: Backend,
{
    fn run(
        &mut self,
        fences: &mut Vec<B::Fence>,
        families: &mut HashMap<QueueFamilyId, Vec<B::CommandQueue>>,
        factory: &mut Factory<B>,
        scene: &mut Scene<B, T, Y>,
    ) -> usize {
        self.0.run(families, factory, scene, fences)
    }

    fn dispose(self, factory: &mut Factory<B>, scene: &mut Scene<B, T, Y>) -> Backbuffer<B> {
        Graph::dispose(self.0, factory, scene);
        unimplemented!()
    }
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "gl", feature = "vulkan")))]
pub fn run<G, F, T, Y>(_: G, _: F)
where
    G: FnOnce(image::Kind, Format, &mut XfgGraphBuilder<Back, T, Y>) -> ImageId,
    F: FnOnce(&mut Scene<Back, T, Y>, &mut Factory<Back>),
{
    env_logger::init();
    error!(
        "You need to enable the native API feature (vulkan/metal/dx12/gl) in order to run example"
    );
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "gl", feature = "vulkan"))]
#[deny(dead_code)]
pub fn run<G, F, T: 'static, Y: 'static>(graph: G, fill: F)
where
    G: FnOnce(image::Kind, Format, &mut XfgGraphBuilder<Back, T, Y>) -> ImageId,
    F: FnOnce(&mut Scene<Back, T, Y>, &mut Factory<Back>),
{
    use std::iter::once;

    env_logger::init();
    let mut events_loop = EventsLoop::new();

    let wb = WindowBuilder::new()
        .with_dimensions(640, 480)
        .with_title("flat".to_string());

    let window = wb.build(&events_loop).unwrap();

    events_loop.poll_events(|_| ());

    let (mut factory, mut render) = gfx::init::<Back, XfgGraph<Back, T, Y>, _, _>(
            gfx::FirstAdapter,
            gfx::queue_picker(|families| families.iter().map(|family| (family, 1)).collect()),
            gfx::MemoryConfig::default(),
    ).unwrap();
    info!("Device features: {:#?}", factory.features());
    info!("Device limits: {:#?}", factory.limits());

    let target = render.add_target(&window, &mut factory);

    events_loop.poll_events(|_| ());

    let mut scene = Scene {
        objects: Vec::new(),
        ambient: AmbientLight([0.0, 0.0, 0.0]),
        lights: Vec::new(),
        camera: Camera {
            transform: Matrix4::identity(),
            projection: Matrix4::identity(),
        },
        other: None,
    };

    events_loop.poll_events(|_| ());

    // fill scene
    fill(&mut scene, &mut factory);

    events_loop.poll_events(|_| ());

    render
        .set_render(
            target,
            &mut factory,
            &mut scene,
            |surface, families, factory, scene| -> Result<_, ::std::io::Error> {
                profile!("render setup");
                let (capabilites, formats, _) = factory.compatibility(&surface);
                let surface_format = formats.map_or(Format::Rgba8Srgb, |formats| {
                    info!("Surface formats: {:#?}", formats);
                    formats
                        .iter()
                        .find(|&format| format.base_format().1 == ChannelType::Srgb)
                        .cloned()
                        .unwrap_or(formats[0])
                });
                info!("Chosen surface format: {:#?}", surface_format);

                let extent = surface.kind().extent();

                info!("Extent: {:#?}", extent);

                scene.camera.projection = PerspectiveFov {
                    fovy: Deg(60.0).into(),
                    aspect: (extent.width as f32) / (extent.height as f32),
                    near: 0.1,
                    far: 2000.0,
                }.into();

                let kind = image::Kind::D2(extent.width.into(), extent.height.into(), 1, 1);

                let mut builder = GraphBuilder::new();
                let surface_id = graph(kind, surface_format, &mut builder);

                profile!("graph building");
                Ok(builder
                    .build(
                        families,
                        create_buffer,
                        create_image,
                        Some(present::PresentBuilder::new(
                            surface_id,
                            surface_format,
                            surface,
                            capabilites,
                        )),
                        factory,
                        scene,
                    )
                    .into())
            },
        )
        .unwrap();

    let start = Instant::now();
    let mut total = 0;
    let total = loop {
        profile!("Frame");
        {
            profile!("Polling");
            events_loop.poll_events(|_| ());
        }
        // Render
        render.run(&mut factory, &mut scene);

        {
            profile!("Counting");
            total += 1;
            if start.elapsed() > Duration::from_millis(10000000) {
                break total;
            }
        }
    };

    let dur = start.elapsed();

    #[cfg(feature = "profile")]
    flame::dump_html(&mut ::std::fs::File::create("profile.html").unwrap()).unwrap();

    events_loop.poll_events(|_| ());
    let dur = (dur.as_secs() as f64 + dur.subsec_nanos() as f64 / 1e9);
    let fps = (total as f64) / dur;
    info!("Run time: {}", dur);
    info!("Total frames rendered: {}", total);
    info!("Average FPS: {}", fps);

    events_loop.poll_events(|_| ());

    // // TODO: Dispose everything properly.
    ::std::process::exit(0);
    // is_send_sync::<back::Device>();
}

fn bindings_to_range(
    multiplier: usize,
    bindings: &[DescriptorSetLayoutBinding],
) -> Vec<DescriptorRangeDesc> {
    let mut ranges = Vec::new();

    for binding in bindings {
        let index = binding.ty as usize;
        while ranges.len() <= index {
            ranges.push(DescriptorRangeDesc {
                ty: DescriptorType::Sampler,
                count: 0,
            });
        }

        let ref mut range = ranges[index];
        range.ty = binding.ty;
        range.count += binding.count * multiplier;
    }

    ranges
}

fn create_image<B, T, Y>(
    kind: image::Kind,
    format: Format,
    usage: image::Usage,
    factory: &mut Factory<B>,
    _: &mut Scene<B, T, Y>,
) -> Image<B>
where
    B: Backend,
{
    factory
        .create_image(
            kind,
            1,
            format,
            image::Tiling::Optimal,
            image::StorageFlags::empty(),
            usage,
            Properties::DEVICE_LOCAL,
        )
        .unwrap()
}

fn create_buffer<B, T, Y>(
    _: u64,
    _: buffer::Usage,
    _: &mut Factory<B>,
    _: &mut Scene<B, T, Y>,
) -> Buffer<B>
where
    B: Backend,
{
    unimplemented!()
}
