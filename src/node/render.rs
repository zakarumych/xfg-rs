use std::{
    borrow::Borrow, iter::{empty, once, Empty}, ops::{Index, Range},
};

use hal::{
    buffer,
    command::{
        ClearColor, ClearValue, CommandBuffer, CommandBufferFlags, MultiShot, OneShot, Primary,
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
        Rect, ShaderStageFlags, StencilTest, VertexBufferDesc, Viewport,
    },
    queue::{CommandQueue, Graphics, RawCommandQueue, RawSubmission}, Backend, Device, Primitive,
};

use relevant::Relevant;

use smallvec::SmallVec;

use node::{build::NodeBuilder, BufferInfo, ImageInfo, Node, NodeDesc};

/// Set layout
#[derive(Clone, Debug, Default)]
pub struct SetLayout {
    pub bindings: Vec<DescriptorSetLayoutBinding>,
}

/// Pipeline layout
#[derive(Clone, Debug)]
pub struct Layout {
    pub sets: Vec<SetLayout>,
    pub push_constants: Vec<(ShaderStageFlags, Range<u32>)>,
}

/// Pipeline info
#[derive(Clone, Debug)]
pub struct Pipeline {
    pub layout: usize,
    pub vertices: Vec<(Vec<Element<Format>>, ElemStride)>,
    pub colors: Vec<ColorBlendDesc>,
    pub depth_stencil: Option<DepthStencilDesc>,
}

/// Render pass desc.
pub trait RenderPassDesc<B: Backend>: Send + Sync + Sized + 'static {
    /// Name of this pass.
    fn name() -> &'static str;

    /// Number of images to sample.
    fn sampled() -> usize {
        0
    }

    /// Number of images to use as storage.
    fn storage() -> usize {
        0
    }

    /// Number of color output images.
    fn colors() -> usize;

    /// Is depth image used.
    fn depth() -> bool {
        false
    }

    /// Pipeline layouts
    fn layouts() -> Vec<Layout> {
        vec![Layout {
            sets: Vec::new(),
            push_constants: Vec::new(),
        }]
    }

    /// Graphics pipelines
    fn pipelines() -> Vec<Pipeline> {
        vec![Pipeline {
            layout: 0,
            vertices: Vec::new(),
            colors: (0..Self::colors())
                .map(|_| ColorBlendDesc(ColorMask::ALL, BlendState::ALPHA))
                .collect(),
            depth_stencil: if Self::depth() {
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
        }]
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
    fn load_shader_sets<'a>(
        storage: &'a mut Vec<B::ShaderModule>,
        device: &mut D,
        aux: &mut T,
    ) -> Vec<GraphicsShaderSet<'a, B>>;

    /// Build pass instance.
    fn build<I>(sampled: I, storage: I, device: &mut D, aux: &mut T) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<B::ImageView>;

    /// Prepare to record drawing commands.
    fn prepare<A, S>(
        &mut self,
        sets: &A,
        cbuf: &mut CommandBuffer<B, Graphics>,
        device: &mut D,
        aux: &T,
    ) where
        A: Index<usize>,
        A::Output: Index<usize, Output = S>,
        S: Borrow<B::DescriptorSetLayout>;

    /// Record drawing commands to the command buffer provided.
    fn draw<L, P>(
        &mut self,
        layouts: &L,
        pipelines: &P,
        encoder: RenderPassInlineEncoder<B, Primary>,
        aux: &T,
    ) where
        L: Index<usize>,
        L::Output: Borrow<B::PipelineLayout>,
        P: Index<usize>,
        P::Output: Borrow<B::GraphicsPipeline>;

    /// Dispose of the pass.
    fn dispose(self, device: &mut D, aux: &mut T);
}

/// Render pass node.
pub struct RenderPassNode<B: Backend, R> {
    relevant: Relevant,

    extent: Extent,

    render_pass: B::RenderPass,
    pipeline_layouts: Vec<B::PipelineLayout>,
    set_layouts: Vec<Vec<B::DescriptorSetLayout>>,
    graphics_pipelines: Vec<B::GraphicsPipeline>,

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
                PipelineStage::COLOR_ATTACHMENT_OUTPUT,
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
                PipelineStage::EARLY_FRAGMENT_TESTS | PipelineStage::LATE_FRAGMENT_TESTS,
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

        let color_info = |index| &images[R::sampled() + R::storage() + index];
        let depth_info = || &images[R::sampled() + R::storage() + R::colors()];

        let render_pass: B::RenderPass = {
            let attachments = (0..R::colors())
                .map(|index| Attachment {
                    format: Some(color_info(index).format),
                    ops: AttachmentOps {
                        load: if color_info(index).clear.is_some() {
                            AttachmentLoadOp::Clear
                        } else {
                            AttachmentLoadOp::Load
                        },
                        store: AttachmentStoreOp::Store,
                    },
                    stencil_ops: AttachmentOps::DONT_CARE,
                    layouts: {
                        let layout = color_info(index).layout;
                        let from = if color_info(index).clear.is_some() {
                            image::Layout::Undefined
                        } else {
                            layout
                        };
                        from..layout
                    },
                    samples: 1,
                })
                .chain(if R::depth() {
                    Some(Attachment {
                        format: Some(depth_info().format),
                        ops: AttachmentOps {
                            load: if depth_info().clear.is_some() {
                                AttachmentLoadOp::Clear
                            } else {
                                AttachmentLoadOp::Load
                            },
                            store: AttachmentStoreOp::Store,
                        },
                        stencil_ops: AttachmentOps::DONT_CARE,
                        layouts: {
                            let layout = depth_info().layout;
                            let from = if depth_info().clear.is_some() {
                                image::Layout::Undefined
                            } else {
                                layout
                            };
                            from..layout
                        },
                        samples: 1,
                    })
                } else {
                    None
                });

            let colors = (0..R::colors())
                .map(|index| (index, color_info(index).layout))
                .collect::<Vec<_>>();
            let depth = if R::depth() {
                Some((R::colors(), depth_info().layout))
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

        trace!("Collect clears for '{}'", R::name());

        let clears = (0..R::colors())
            .map(|index| {
                color_info(index)
                    .clear
                    .unwrap_or(ClearValue::Color(ClearColor::Float([0.3, 0.7, 0.9, 1.0])))
            })
            .chain(if R::depth() { depth_info().clear } else { None })
            .collect();

        trace!("Create views for '{}'", R::name());

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

        trace!("Creating layouts for '{}'", R::name());

        let (pipeline_layouts, set_layouts): (Vec<_>, Vec<_>) = R::layouts()
            .into_iter()
            .map(|layout| {
                let set_layouts = layout
                    .sets
                    .into_iter()
                    .map(|set| device.create_descriptor_set_layout(set.bindings))
                    .collect::<Vec<_>>();
                let pipeline_layout =
                    device.create_pipeline_layout(&set_layouts, layout.push_constants);
                (pipeline_layout, set_layouts)
            })
            .unzip();

        trace!("Creating graphics pipelines for '{}'", R::name());

        let graphics_pipelines = {
            let mut shaders = Vec::new();

            let pipelines = R::pipelines();

            let descs = pipelines
                .iter()
                .enumerate()
                .zip(R::load_shader_sets(&mut shaders, device, aux))
                .map(|((index, pipeline), shader_set)| {
                    assert_eq!(pipeline.colors.len(), R::colors());
                    assert_eq!(pipeline.depth_stencil.is_some(), R::depth());

                    let mut vertex_buffers = Vec::new();
                    let mut attributes = Vec::new();

                    for &(ref elemets, stride) in &pipeline.vertices {
                        push_vertex_desc(elemets, stride, &mut vertex_buffers, &mut attributes);
                    }

                    GraphicsPipelineDesc {
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
                            targets: pipeline.colors.clone(),
                        },
                        depth_stencil: pipeline.depth_stencil,
                        multisampling: None,
                        baked_states: BakedStates {
                            viewport: Some(Viewport {
                                rect,
                                depth: 0.0..1.0,
                            }),
                            scissor: Some(rect),
                            blend_color: None,
                            depth_bounds: None,
                        },
                        layout: &pipeline_layouts[pipeline.layout],
                        subpass: Subpass {
                            index: 0,
                            main_pass: &render_pass,
                        },
                        flags: if index == 0 && pipelines.len() > 1 {
                            PipelineCreationFlags::ALLOW_DERIVATIVES
                        } else {
                            PipelineCreationFlags::empty()
                        },
                        parent: if index == 0 {
                            BasePipeline::None
                        } else {
                            BasePipeline::Index(0)
                        },
                    }
                });

            let pipelines = device
                .create_graphics_pipelines(descs)
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            trace!("Graphics pipeline created for '{}'", R::name());
            pipelines
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
            info.barriers
                .acquire
                .as_ref()
                .map(|barrier| (barrier, info.image.borrow(), info.format.aspects()))
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
                info.barriers
                    .release
                    .as_ref()
                    .map(|barrier| (barrier, info.image.borrow(), info.format.aspects()))
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
            .create_framebuffer(
                &render_pass,
                views.iter().skip(R::sampled() + R::storage()),
                extent,
            )
            .unwrap();

        let pass = R::build(
            &views[..R::sampled()],
            &views[R::sampled()..R::sampled() + R::storage()],
            device,
            aux,
        );

        RenderPassNode {
            relevant: Relevant,
            extent,
            render_pass,
            set_layouts,
            pipeline_layouts,
            graphics_pipelines,
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

        self.pool.reset();
        let mut cbuf = self.pool.acquire_command_buffer::<OneShot>(false);
        {
            profile!("Render pass prepare");
            self.pass.prepare(&self.set_layouts, &mut cbuf, device, aux);
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
                self.pass.draw(
                    &self.pipeline_layouts,
                    &self.graphics_pipelines,
                    encoder,
                    aux,
                );
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
        // | PipelineStage::DOMAIN_SHADER
        // | PipelineStage::HULL_SHADER
        // | PipelineStage::GEOMETRY_SHADER
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
