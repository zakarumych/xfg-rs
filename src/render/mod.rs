
use std::{borrow::Borrow, iter::{empty, Empty}, ops::Range};

use hal::{ Backend, Device, buffer, image, Primitive, 
    command::{CommandBuffer, OneShot, MultiShot, Primary, Submit},
    format::{Format, Swizzle},
    memory::{Dependencies, Barrier},
    image::Extent,
    queue::Graphics,
    pass::{Attachment, AttachmentOps, AttachmentLoadOp, AttachmentStoreOp, SubpassDesc, Subpass, SubpassDependency},
    pool::{CommandPool, CommandPoolCreateFlags, RawCommandPool},
    pso::{
        Element, ElemStride, VertexBufferDesc, PipelineStage,
        AttributeDesc, InputAssemblerDesc, PrimitiveRestart,
        ColorBlendDesc, ColorMask, BlendState, DepthTest, DepthStencilDesc,
        StencilTest, Comparison, BakedStates, BasePipeline, Rasterizer,
        PipelineCreationFlags, DescriptorSetLayoutBinding, BufferIndex,
        GraphicsShaderSet, BlendDesc, GraphicsPipelineDesc, ShaderStageFlags,
    },
};

use relevant::Relevant;

use smallvec::SmallVec;

use node::{Node, NodeDesc, Submittables, EitherSubmit, BufferInfo, ImageInfo};

/// Render pass desc.
pub trait RenderPassDesc: Send + Sync + Sized + 'static {
    /// Name of this pass.
    fn name() -> &'static str;

    /// Number of images to sample.
    fn sampled() -> usize;

    /// Number of color output images.
    fn colors() -> usize;

    /// Is depth image used.
    fn depth() -> bool;

    /// Format of vertices.
    fn vertices() -> &'static [(&'static [Element<Format>], ElemStride)];

    /// Bindings for the pass.
    fn bindings() -> &'static [DescriptorSetLayoutBinding];
}

/// Render pass.
pub trait RenderPass<B: Backend, D, T>: RenderPassDesc {

    /// Load shader set.
    /// This function should create required shader modules and fill `GraphicsShaderSet` structure.
    ///
    /// # Parameters
    ///
    /// `storage`   - vector where this function can store loaded modules to give them required lifetime.
    ///
    /// `device`    - `Device<B>` implementation. `B::Device` or wrapper.
    ///
    /// `aux`       - auxiliary data container. May be anything the implementation desires.
    ///
    fn load_shader_set<'a>(storage: &'a mut Vec<B::ShaderModule>, device: &mut D, aux: &mut T) -> GraphicsShaderSet<'a, B>;

    /// Build pass instance.
    fn build(
        frames: usize,
        device: &mut D,
        aux: &mut T,
    ) -> Self;

    /// Record drawing commands to the command buffer provided.
    fn run<I>(&mut self, sampled: I, frame: usize, device: &mut D, aux: &T, &mut CommandBuffer<B, Graphics>)
    where
        I: IntoIterator,
        I::Item: Borrow<B::ImageView>,
    ;

    /// Dispose of the pass.
    fn dispose(self, device: &mut D, aux: &mut T);
}

struct Resources<B: Backend> {
    views: Vec<B::ImageView>,
    framebuffer: B::Framebuffer,
    pool: CommandPool<B, Graphics>,
    barriers_pool: CommandPool<B, Graphics>,
    acquire: Option<Submit<B, Graphics, MultiShot, Primary>>,
    release: Option<Submit<B, Graphics, MultiShot, Primary>>,
}

/// Render pass node.
pub struct RenderPassNode<B: Backend, R> {
    relevant: Relevant,
    resources: Vec<Resources<B>>,
    render_pass: B::RenderPass,
    pipeline_layout: B::PipelineLayout,
    graphics_pipeline: B::GraphicsPipeline,
    pass: R,
}

/// Overall description for node.
impl<B, R> NodeDesc for RenderPassNode<B, R>
where
    B: Backend,
    R: RenderPassDesc,
{
    type Buffers = Empty<(buffer::Usage, buffer::State, PipelineStage)>;

    type Images = SmallVec<[(image::Usage, image::State, PipelineStage); 16]>;

    type Capability = Graphics;

    fn name() -> &'static str {
        R::name()
    }

    fn buffers() -> Self::Buffers {
        empty()
    }

    fn images() -> Self::Images {
        let sampled = (0 .. R::sampled()).map(|_| (image::Usage::SAMPLED, (image::Access::SHADER_READ, image::Layout::ShaderReadOnlyOptimal), all_graphics_shaders_stages()));
        let colors = (0 .. R::colors()).map(|_| (image::Usage::COLOR_ATTACHMENT, (image::Access::COLOR_ATTACHMENT_READ | image::Access::COLOR_ATTACHMENT_WRITE, image::Layout::ColorAttachmentOptimal), all_graphics_shaders_stages()));
        let depth = if R::depth() { Some((image::Usage::DEPTH_STENCIL_ATTACHMENT, (image::Access::DEPTH_STENCIL_ATTACHMENT_READ | image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE, image::Layout::DepthStencilAttachmentOptimal), all_graphics_shaders_stages())) } else { None };

        sampled.chain(colors).chain(depth).collect()
    }
}

impl<'a, B, D, T, R> Submittables<'a, B, D, T> for RenderPassNode<B, R>
where
    B: Backend,
    R: RenderPassDesc,
{
    type Submittable = EitherSubmit<'a, B, Graphics>;
    type IntoIter = SmallVec<[EitherSubmit<'a, B, Graphics>; 3]>;
}

impl<B, D, T, R> Node<B, D, T> for RenderPassNode<B, R>
where
    B: Backend,
    D: Device<B>,
    R: RenderPass<B, D, T>,
{
    fn build<F>(
        buffers: Vec<BufferInfo<B>>,
        images: Vec<ImageInfo<B>>,
        frames: usize,
        mut pools: F,
        device: &mut D,
        aux: &mut T,
    ) -> Self
    where
        F: FnMut(&mut D, CommandPoolCreateFlags) -> CommandPool<B, Self::Capability>,
    {
        trace!("Creating RenderPass instance for '{}'", R::name());
        let render_pass: B::RenderPass = {
            let attachments = (0 .. R::colors()).map(|index| Attachment {
                format: Some(images[R::sampled() + index].format),
                ops: AttachmentOps {
                    load: AttachmentLoadOp::Load,
                    store: AttachmentStoreOp::Store,
                },
                stencil_ops: AttachmentOps::DONT_CARE,
                layouts: {
                    let layout = images[index + R::sampled()].layout;
                    layout .. layout
                },
                samples: 1,
            }).chain( if R::depth() {
                Some(Attachment {
                    format: Some(images[R::sampled() + R::colors()].format),
                    ops: AttachmentOps {
                        load: AttachmentLoadOp::Load,
                        store: AttachmentStoreOp::Store,
                    },
                    stencil_ops: AttachmentOps::DONT_CARE,
                    layouts: {
                        let layout = images[R::sampled() + R::colors()].layout;
                        layout .. layout
                    },
                    samples: 1,
                })
            } else {
                None
            });

            let colors = (0..R::colors()).map(|index| (index, images[index + R::sampled()].layout)).collect::<Vec<_>>();
            let depth = if R::depth() { Some((R::colors(), images[R::colors() + R::sampled()].layout)) } else { None };
            let subpass = SubpassDesc {
                colors: &colors,
                depth_stencil: depth.as_ref(),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let result = device.create_render_pass(
                attachments,
                Some(subpass),
                empty::<SubpassDependency>(),
            );

            trace!("RenderPass instance created for '{}'", R::name());
            result
        };

        trace!("Creating graphics pipeline for '{}'", R::name());
        let descriptor_set_layout = device.create_descriptor_set_layout(R::bindings());
        let pipeline_layout = device.create_pipeline_layout(Some(&descriptor_set_layout), empty::<(ShaderStageFlags, Range<u32>)>());

        let graphics_pipeline = {
            let mut shaders = Vec::new();
            let shader_set = R::load_shader_set(&mut shaders, device, aux);

            let mut vertex_buffers = Vec::new();
            let mut attributes = Vec::new();

            for &(elemets, stride) in R::vertices() {
                push_vertex_desc(elemets, stride, &mut vertex_buffers, &mut attributes);
            }

            let result = device.create_graphics_pipeline(&GraphicsPipelineDesc {
                shaders: shader_set,
                rasterizer: Rasterizer::FILL,
                vertex_buffers,
                attributes,
                input_assembler: InputAssemblerDesc {
                    primitive: Primitive::TriangleList,
                    primitive_restart: PrimitiveRestart::Disabled,
                },
                blender: BlendDesc {
                    logic_op: None,
                    targets: (0 .. R::colors()).map(|_| ColorBlendDesc(ColorMask::ALL, BlendState::ALPHA)).collect(),
                },
                depth_stencil: if R::depth() {
                    Some(DepthStencilDesc {
                        depth: DepthTest::On {
                            fun: Comparison::LessEqual,
                            write: true,
                        },
                        depth_bounds: false,
                        stencil: StencilTest::Off,
                    })
                } else {
                    None
                },
                multisampling: None,
                baked_states: BakedStates::default(),
                layout: &pipeline_layout,
                subpass: Subpass {
                    index: 0,
                    main_pass: &render_pass,
                },
                flags: PipelineCreationFlags::empty(),
                parent: BasePipeline::None,
            }).unwrap();
            trace!("Graphics pipeline created for '{}'", R::name());
            result
        };

        let with_acquire = buffers.iter().any(|buffer| buffer.barriers.acquire.is_some());
        let with_release = buffers.iter().any(|buffer| buffer.barriers.release.is_some());

        RenderPassNode {
            resources: (0 .. frames).map(|index| {
                let mut extent = None;
                let views = images.iter().enumerate().map(|(i, info)| {
                    if i >= R::sampled() {
                        // This is color or depth attachment.
                        assert!(extent.map_or(true, |e| e == info.kind.extent()), "All attachments must have same `Extent`");
                        extent = Some(info.kind.extent());
                    }
                    device.create_image_view(
                        info.images[index],
                        match info.kind {
                            image::Kind::D1(_, _) => image::ViewKind::D1,
                            image::Kind::D2(_, _, _, _) => image::ViewKind::D2,
                            image::Kind::D3(_, _, _) => image::ViewKind::D3,
                        },
                        info.format,
                        Swizzle::NO,
                        image::SubresourceRange {
                            aspects: info.format.aspects(),
                            levels: 0 .. 1,
                            layers: 0 .. 1,
                        },
                    ).unwrap()
                }).collect::<Vec<_>>();

                let extent = extent.unwrap_or(Extent { width: 0, height: 0, depth: 0 });


                let mut barriers_pool = pools(device, CommandPoolCreateFlags::empty());

                let acquire = if with_acquire {
                    let mut cbuf = barriers_pool.acquire_command_buffer::<MultiShot>(false);
                    for (barrier, buffer) in buffers.iter().filter_map(|info| info.barriers.acquire.as_ref().map(|barrier| (barrier, info.buffers[index]))) {
                        cbuf.pipeline_barrier(
                            barrier.start.1 .. barrier.end.1,
                            Dependencies::empty(),
                            Some(Barrier::Buffer {
                                states: barrier.start.0 .. barrier.end.0,
                                target: buffer,
                            })
                        );
                    }

                    for (barrier, image, aspects) in images.iter().filter_map(|info| info.barriers.acquire.as_ref().map(|barrier| (barrier, info.images[index], info.format.aspects()))) {
                        cbuf.pipeline_barrier(
                            barrier.start.1 .. barrier.end.1,
                            Dependencies::empty(),
                            Some(Barrier::Image {
                                states: barrier.start.0 .. barrier.end.0,
                                target: image,
                                range: image::SubresourceRange {
                                    aspects,
                                    levels: 0..1,
                                    layers: 0..1,
                                }
                            })
                        );
                    }
                    Some(cbuf.finish())
                } else {
                    None
                };

                let release = if with_release {
                    let mut cbuf = barriers_pool.acquire_command_buffer::<MultiShot>(false);
                    for (barrier, buffer) in buffers.iter().filter_map(|info| info.barriers.release.as_ref().map(|barrier| (barrier, info.buffers[index]))) {
                        cbuf.pipeline_barrier(
                            barrier.start.1 .. barrier.end.1,
                            Dependencies::empty(),
                            Some(Barrier::Buffer {
                                states: barrier.start.0 .. barrier.end.0,
                                target: buffer,
                            })
                        );
                    }

                    for (barrier, image, aspects) in images.iter().filter_map(|info| info.barriers.release.as_ref().map(|barrier| (barrier, info.images[index], info.format.aspects()))) {
                        cbuf.pipeline_barrier(
                            barrier.start.1 .. barrier.end.1,
                            Dependencies::empty(),
                            Some(Barrier::Image {
                                states: barrier.start.0 .. barrier.end.0,
                                target: image,
                                range: image::SubresourceRange {
                                    aspects,
                                    levels: 0..1,
                                    layers: 0..1,
                                }
                            })
                        );
                    }
                    Some(cbuf.finish())
                } else {
                    None
                };

                Resources {
                    framebuffer: device.create_framebuffer(&render_pass, views.iter().skip(R::sampled()), extent).unwrap(),
                    views,
                    pool: pools(device, CommandPoolCreateFlags::empty()),
                    barriers_pool,
                    acquire,
                    release,
                }
            }).collect(),
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            pass: R::build(frames, device, aux),
            relevant: Relevant,
        }
    }

    fn run<'a>(
        &'a mut self,
        frame: usize,
        device: &mut D,
        aux: &'a T,
    ) -> SmallVec<[EitherSubmit<'a, B, Graphics>; 3]>
    {
        let ref mut resources = self.resources[frame];
        let mut cbuf = resources.pool.acquire_command_buffer::<OneShot>(false);
        self.pass.run(&resources.views[..R::sampled()], frame, device, aux, &mut cbuf);
        let mut main = Some(cbuf.finish().into());

        let acquire = resources.acquire.as_ref().map(Into::into);
        let release = resources.release.as_ref().map(Into::into);
        
        acquire
            .into_iter()
            .chain(main)
            .chain(release)
            .collect()
    }

    fn dispose(self, device: &mut D, aux: &mut T) {
        self.pass.dispose(device, aux);

        for mut resources in self.resources {
            drop(resources.acquire);
            drop(resources.release);
            resources.barriers_pool.reset();
            resources.pool.reset();
            device.destroy_command_pool(resources.barriers_pool.into_raw());
            device.destroy_command_pool(resources.pool.into_raw());
        }

        self.relevant.dispose();
    }
}

fn all_graphics_shaders_stages() -> PipelineStage {
    PipelineStage::VERTEX_SHADER |
    PipelineStage::DOMAIN_SHADER |
    PipelineStage::HULL_SHADER |
    PipelineStage::GEOMETRY_SHADER |
    PipelineStage::FRAGMENT_SHADER
}

fn push_vertex_desc(
    elements: &[Element<Format>],
    stride: ElemStride,
    vertex_buffers: &mut Vec<VertexBufferDesc>,
    attributes: &mut Vec<AttributeDesc>,
) {
    let index = vertex_buffers.len() as BufferIndex;

    vertex_buffers
        .push(VertexBufferDesc { binding: 0, stride, rate: 0 });

    let mut location = attributes
        .last()
        .map(|a| a.location + 1)
        .unwrap_or(0);
    for &element in elements {
        attributes.push(AttributeDesc {
            location,
            binding: index,
            element,
        });
        location += 1;
    }
}
