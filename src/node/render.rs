use std::{
    borrow::Borrow, iter::{empty, once, Empty}, ops::Range,
};

use hal::{
    buffer,
    command::{
        ClearValue, ClearColor, CommandBuffer, CommandBufferFlags, MultiShot, OneShot, Primary,
        RawCommandBuffer, RawLevel, RenderPassInlineEncoder, Submit, Submittable,
    },
    format::{Format, Swizzle}, image, image::Extent, memory::{Barrier, Dependencies},
    pass::{
        Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDependency,
        SubpassDesc,
    },
    pool::{CommandPool, CommandPoolCreateFlags, RawCommandPool},
    pso::{
        AttributeDesc, BakedStates, BasePipeline, BlendDesc, BlendState, BufferIndex,
        ColorBlendDesc, ColorMask, Comparison, DepthStencilDesc, DepthTest,
        DescriptorSetLayoutBinding, ElemStride, Element, GraphicsPipelineDesc, GraphicsShaderSet,
        InputAssemblerDesc, PipelineCreationFlags, PipelineStage, PrimitiveRestart, Rasterizer,
        Rect, Viewport, ShaderStageFlags, StencilTest, VertexBufferDesc,
    },
    queue::{CommandQueue, Graphics, RawCommandQueue, RawSubmission}, Backend, Device, Primitive,
};

use relevant::Relevant;

use smallvec::SmallVec;

use node::{build::NodeBuilder, BufferInfo, ImageInfo, Node, NodeDesc};

/// Render pass desc.
pub trait RenderPassDesc<B: Backend>: Send + Sync + Sized + 'static {
    /// Name of this pass.
    fn name() -> &'static str;

    /// Number of images to sample.
    fn sampled() -> usize { 0 }

    /// Number of images to use as storage.
    fn storage() -> usize { 0 }

    /// Number of color output images.
    fn colors() -> usize;

    /// Get color blend description for color target by index.
    fn color_blend(_index: usize) -> ColorBlendDesc {
        ColorBlendDesc(ColorMask::ALL, BlendState::ALPHA)
    }

    /// Is depth image used.
    fn depth() -> bool { false }

    /// Format of vertices.
    fn vertices() -> &'static [(&'static [Element<Format>], ElemStride)] {
        &[]
    }

    /// Bindings for the pass.
    fn bindings() -> &'static [DescriptorSetLayoutBinding] {
        &[]
    }

    /// Create `NodeBuilder` for this node.
    fn builder() -> NodeBuilder<RenderPassNode<B, Self>> {
        RenderPassNode::builder()
    }
}

/// Render pass.
pub trait RenderPass<B, D, T>: RenderPassDesc<B>
where
    B: Backend,
    D: Device<B>,
{
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
    fn load_shader_set<'a>(
        storage: &'a mut Vec<B::ShaderModule>,
        device: &mut D,
        aux: &mut T,
    ) -> GraphicsShaderSet<'a, B>;

    /// Build pass instance.
    fn build<I>(
        sampled: I,
        storage: I,
        set: &B::DescriptorSetLayout,
        device: &mut D,
        aux: &mut T
    ) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<B::ImageView>;

    /// Prepare to record drawing commands.
    fn prepare(
        &mut self,
        set: &B::DescriptorSetLayout,
        cbuf: &mut CommandBuffer<B, Graphics>,
        device: &mut D,
        aux: &T,
    );

    /// Record drawing commands to the command buffer provided.
    fn draw(
        &mut self,
        pipeline: &B::PipelineLayout,
        encoder: RenderPassInlineEncoder<B, Primary>,
        aux: &T,
    );

    /// Dispose of the pass.
    fn dispose(self, device: &mut D, aux: &mut T);
}

/// Render pass node.
pub struct RenderPassNode<B: Backend, R> {
    relevant: Relevant,

    extent: Extent,

    render_pass: B::RenderPass,
    set_layout: B::DescriptorSetLayout,
    pipeline_layout: B::PipelineLayout,
    graphics_pipeline: B::GraphicsPipeline,

    views: Vec<B::ImageView>,
    framebuffer: B::Framebuffer,
    clears: Vec<ClearValue>,

    pool: CommandPool<B, Graphics>,
    static_pool: B::CommandPool,
    acquire: B::CommandBuffer,
    release: Option<B::CommandBuffer>,

    pass: R,
}

/// Overall description for node.
impl<B, R> NodeDesc for RenderPassNode<B, R>
where
    B: Backend,
    R: RenderPassDesc<B>,
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
        let sampled = (0..R::sampled()).map(|_| {
            (
                image::Usage::SAMPLED,
                (
                    image::Access::SHADER_READ,
                    image::Layout::ShaderReadOnlyOptimal,
                ),
                all_graphics_shaders_stages(),
            )
        });
        let storage = (0..R::storage()).map(|_| {
            (
                image::Usage::STORAGE,
                (
                    image::Access::SHADER_READ,
                    image::Layout::ShaderReadOnlyOptimal,
                ),
                all_graphics_shaders_stages(),
            )
        });
        let colors = (0..R::colors()).map(|_| {
            (
                image::Usage::COLOR_ATTACHMENT,
                (
                    image::Access::COLOR_ATTACHMENT_READ | image::Access::COLOR_ATTACHMENT_WRITE,
                    image::Layout::ColorAttachmentOptimal,
                ),
                all_graphics_shaders_stages(),
            )
        });
        let depth = if R::depth() {
            Some((
                image::Usage::DEPTH_STENCIL_ATTACHMENT,
                (
                    image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                        | image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    image::Layout::DepthStencilAttachmentOptimal,
                ),
                all_graphics_shaders_stages(),
            ))
        } else {
            None
        };

        sampled.chain(storage).chain(colors).chain(depth).collect()
    }
}

impl<B, D, T, R> Node<B, D, T> for RenderPassNode<B, R>
where
    B: Backend,
    D: Device<B>,
    R: RenderPass<B, D, T>,
{
    fn build<F, U, I>(
        buffers: Vec<BufferInfo<U>>,
        images: Vec<ImageInfo<I>>,
        mut pools: F,
        device: &mut D,
        aux: &mut T,
    ) -> Self
    where
        F: FnMut(&mut D, CommandPoolCreateFlags) -> CommandPool<B, Self::Capability>,
        U: Borrow<B::Buffer>,
        I: Borrow<B::Image>,
    {
        trace!("Creating RenderPass instance for '{}'", R::name());

        let render_pass: B::RenderPass = {
            let attachments = (0..R::colors())
                .map(|index| Attachment {
                    format: Some(images[R::sampled() + R::storage() + index].format),
                    ops: AttachmentOps {
                        load: if images[R::sampled() + R::storage() + index].clear.is_some() {
                            AttachmentLoadOp::Clear
                        } else {
                            AttachmentLoadOp::Load
                        },
                        store: AttachmentStoreOp::Store,
                    },
                    stencil_ops: AttachmentOps::DONT_CARE,
                    layouts: {
                        let layout = images[R::sampled() + R::storage() + index].layout;
                        layout..layout
                    },
                    samples: 1,
                })
                .chain(if R::depth() {
                    Some(Attachment {
                        format: Some(images[R::sampled() + R::storage() + R::colors()].format),
                        ops: AttachmentOps {
                            load: if images[R::sampled() + R::storage() + R::colors()].clear.is_some() {
                                AttachmentLoadOp::Clear
                            } else {
                                AttachmentLoadOp::Load
                            },
                            store: AttachmentStoreOp::Store,
                        },
                        stencil_ops: AttachmentOps::DONT_CARE,
                        layouts: {
                            let layout = images[R::sampled() + R::storage() + R::colors()].layout;
                            layout..layout
                        },
                        samples: 1,
                    })
                } else {
                    None
                });

            let colors = (0..R::colors())
                .map(|index| (index, images[R::sampled() + R::storage() + index].layout))
                .collect::<Vec<_>>();
            let depth = if R::depth() {
                Some((R::colors(), images[R::sampled() + R::storage() + R::colors()].layout))
            } else {
                None
            };

            let subpass = SubpassDesc {
                colors: &colors,
                depth_stencil: depth.as_ref(),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let result =
                device.create_render_pass(attachments, Some(subpass), empty::<SubpassDependency>());

            trace!("RenderPass instance created for '{}'", R::name());
            result
        };

        let clears = (0 .. R::colors() + R::depth() as usize).map(|index| {
            images[R::sampled() + R::storage() + index].clear.unwrap_or(ClearValue::Color(ClearColor::Float([0.3, 0.7, 0.9, 1.0])))
        }).collect();

        trace!("Creating graphics pipeline for '{}'", R::name());
        let set_layout = device.create_descriptor_set_layout(R::bindings());
        let pipeline_layout = device
            .create_pipeline_layout(Some(&set_layout), empty::<(ShaderStageFlags, Range<u32>)>());

        let mut extent = None;

        let views = images
            .iter()
            .enumerate()
            .map(|(i, info)| {
                if i >= R::sampled() + R::storage() {
                    // This is color or depth attachment.
                    assert!(
                        extent.map_or(true, |e| e == info.kind.extent()),
                        "All attachments must have same `Extent`"
                    );
                    extent = Some(info.kind.extent());
                }

                device
                    .create_image_view(
                        info.image.borrow(),
                        match info.kind {
                            image::Kind::D1(_, _) => image::ViewKind::D1,
                            image::Kind::D2(_, _, _, _) => image::ViewKind::D2,
                            image::Kind::D3(_, _, _) => image::ViewKind::D3,
                        },
                        info.format,
                        Swizzle::NO,
                        image::SubresourceRange {
                            aspects: info.format.aspects(),
                            levels: 0..1,
                            layers: 0..1,
                        },
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let extent = extent.unwrap_or(Extent {
            width: 0,
            height: 0,
            depth: 0,
        });

        let rect = Rect {
            x: 0,
            y: 0,
            w: extent.width as _,
            h: extent.height as _,
        };

        let graphics_pipeline = {
            let mut shaders = Vec::new();
            let shader_set = R::load_shader_set(&mut shaders, device, aux);

            let mut vertex_buffers = Vec::new();
            let mut attributes = Vec::new();

            for &(elemets, stride) in R::vertices() {
                push_vertex_desc(elemets, stride, &mut vertex_buffers, &mut attributes);
            }

            let result = device
                .create_graphics_pipeline(&GraphicsPipelineDesc {
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
                        targets: (0..R::colors())
                            .map(|index| R::color_blend(index))
                            .collect(),
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
                    baked_states: BakedStates {
                        viewport: Some(Viewport {rect, depth: 0.0 .. 1.0 }),
                        scissor: Some(rect),
                        blend_color: None,
                        depth_bounds: None,
                    },
                    layout: &pipeline_layout,
                    subpass: Subpass {
                        index: 0,
                        main_pass: &render_pass,
                    },
                    flags: PipelineCreationFlags::empty(),
                    parent: BasePipeline::None,
                })
                .unwrap();
            trace!("Graphics pipeline created for '{}'", R::name());
            result
        };

        let mut static_pool = pools(device, CommandPoolCreateFlags::empty()).into_raw();

        let with_release = buffers.iter().any(|info| info.barriers.release.is_some())
            || images.iter().any(|info| info.barriers.release.is_some());

        let mut static_cbufs = static_pool.allocate(1 + with_release as usize, RawLevel::Primary);

        let mut acquire = static_cbufs.pop().unwrap();
        acquire.begin(CommandBufferFlags::EMPTY, Default::default());

        for (barrier, buffer) in buffers.iter().filter_map(|info| {
            info.barriers
                .acquire
                .as_ref()
                .map(|barrier| (barrier, info.buffer.borrow()))
        }) {
            acquire.pipeline_barrier(
                barrier.start.1..barrier.end.1,
                Dependencies::empty(),
                Some(Barrier::Buffer {
                    states: barrier.start.0..barrier.end.0,
                    target: buffer,
                }),
            );
        }
        for (barrier, image, aspects) in images.iter().filter_map(|info| {
            info.barriers.acquire.as_ref().map(|barrier| {
                (
                    barrier,
                    info.image.borrow(),
                    info.format.aspects(),
                )
            })
        }) {
            acquire.pipeline_barrier(
                barrier.start.1..barrier.end.1,
                Dependencies::empty(),
                Some(Barrier::Image {
                    states: barrier.start.0..barrier.end.0,
                    target: image,
                    range: image::SubresourceRange {
                        aspects,
                        levels: 0..1,
                        layers: 0..1,
                    },
                }),
            );
        }
        acquire.bind_graphics_pipeline(&graphics_pipeline);
        acquire.finish();

        let release = if with_release {
            let mut release = static_cbufs.pop().unwrap();
            release.begin(CommandBufferFlags::EMPTY, Default::default());
            for (barrier, buffer) in buffers.iter().filter_map(|info| {
                info.barriers
                    .release
                    .as_ref()
                    .map(|barrier| (barrier, info.buffer.borrow()))
            }) {
                release.pipeline_barrier(
                    barrier.start.1..barrier.end.1,
                    Dependencies::empty(),
                    Some(Barrier::Buffer {
                        states: barrier.start.0..barrier.end.0,
                        target: buffer,
                    }),
                );
            }

            for (barrier, image, aspects) in images.iter().filter_map(|info| {
                info.barriers.release.as_ref().map(|barrier| {
                    (
                        barrier,
                        info.image.borrow(),
                        info.format.aspects(),
                    )
                })
            }) {
                release.pipeline_barrier(
                    barrier.start.1..barrier.end.1,
                    Dependencies::empty(),
                    Some(Barrier::Image {
                        states: barrier.start.0..barrier.end.0,
                        target: image,
                        range: image::SubresourceRange {
                            aspects,
                            levels: 0..1,
                            layers: 0..1,
                        },
                    }),
                );
            }
            release.finish();
            Some(release)
        } else {
            None
        };

        let framebuffer = device
            .create_framebuffer(&render_pass, views.iter().skip(R::sampled() + R::storage()), extent)
            .unwrap();

        let pass = R::build(&views[..R::sampled()], &views[R::sampled() .. R::sampled() + R::storage()], &set_layout, device, aux);

        RenderPassNode {
            relevant: Relevant,
            extent,
            render_pass,
            set_layout: set_layout,
            pipeline_layout,
            graphics_pipeline,
            pool: pools(device, CommandPoolCreateFlags::empty()),
            static_pool,
            acquire,
            release,
            views,
            framebuffer,
            clears,
            pass,
        }
    }

    #[inline]
    fn run<'a, W, S>(
        &'a mut self,
        wait: W,
        queue: &mut CommandQueue<B, Graphics>,
        signal: S,
        fence: Option<&B::Fence>,
        device: &mut D,
        aux: &'a T,
    ) where
        W: IntoIterator<Item = (&'a B::Semaphore, PipelineStage)>,
        S: IntoIterator<Item = &'a B::Semaphore>,
    {
        profile!("RenderPassNode::run");

        let area = Rect {
            x: 0,
            y: 0,
            w: self.extent.width as u16,
            h: self.extent.height as u16,
        };

        let mut cbuf = self.pool.acquire_command_buffer::<OneShot>(false);

        {
            profile!("bind graphics pipeline");
            cbuf.bind_graphics_pipeline(&self.graphics_pipeline);
        }
        {
            profile!("Render pass prepare");
            self.pass.prepare(&self.set_layout, &mut cbuf, device, aux);
        }
        {
            let encoder = {
                profile!("begin render pass");
                cbuf.begin_render_pass_inline(
                    &self.render_pass,
                    &self.framebuffer,
                    area,
                    &self.clears,
                )
            };
            {
                profile!("Render pass draw");
                self.pass.draw(&self.pipeline_layout, encoder, aux);
            }
        }

        unsafe {
            profile!("Submission");
            queue.as_raw_mut().submit_raw(
                RawSubmission {
                    wait_semaphores: &wait
                        .into_iter()
                        .map(|(semaphore, stage)| (semaphore.borrow(), stage))
                        .collect::<SmallVec<[_; 16]>>(),
                    cmd_buffers: once(&self.acquire)
                        .chain(once(cbuf.finish().into_buffer().as_ref()))
                        .chain(self.release.as_ref()),
                    signal_semaphores: &signal
                        .into_iter()
                        .map(Borrow::borrow)
                        .collect::<SmallVec<[_; 16]>>(),
                },
                fence,
            );
        }
    }

    fn dispose(mut self, device: &mut D, aux: &mut T) {
        self.pass.dispose(device, aux);
        drop(self.acquire);
        drop(self.release);
        self.static_pool.reset();
        self.pool.reset();
        device.destroy_command_pool(self.static_pool);
        device.destroy_command_pool(self.pool.into_raw());
        self.relevant.dispose();
    }
}

fn all_graphics_shaders_stages() -> PipelineStage {
    PipelineStage::VERTEX_SHADER
        | PipelineStage::DOMAIN_SHADER
        | PipelineStage::HULL_SHADER
        | PipelineStage::GEOMETRY_SHADER
        | PipelineStage::FRAGMENT_SHADER
}

fn push_vertex_desc(
    elements: &[Element<Format>],
    stride: ElemStride,
    vertex_buffers: &mut Vec<VertexBufferDesc>,
    attributes: &mut Vec<AttributeDesc>,
) {
    let index = vertex_buffers.len() as BufferIndex;

    vertex_buffers.push(VertexBufferDesc {
        binding: 0,
        stride,
        rate: 0,
    });

    let mut location = attributes.last().map(|a| a.location + 1).unwrap_or(0);
    for &element in elements {
        attributes.push(AttributeDesc {
            location,
            binding: index,
            element,
        });
        location += 1;
    }
}
