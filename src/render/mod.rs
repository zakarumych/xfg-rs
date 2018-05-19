
use std::{borrow::Borrow, iter::{empty, Empty}, ops::Range};

use hal::{ Backend, Device, buffer, image, Primitive, 
    command::{CommandBuffer, Primary, OneShot},
    format::Format,
    image::Extent,
    queue::Graphics,
    pass::{Attachment, AttachmentOps, AttachmentLoadOp, AttachmentStoreOp, SubpassDesc},
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{Element, ElemStride, VertexBufferDesc, PipelineStage, AttributeDesc, InputAssemblerDesc, PrimitiveRestart, ColorBlendDesc, ColorMask, BlendState, DepthTest, DepthStencilDesc, StencilTest, Comparison, BakedStates, Subpass, BasePipeline},
};

use id::{BufferId, ImageId};
use node::{Node, NodeDesc, Submittables, EitherSubmit};

/// Render pass desc.
pub trait RenderPassDesc: Send + Sync + Sized + 'static {
    /// Name of this pass.
    fn name() -> &str;

    /// Number of images to sample.
    fn sampled() -> usize;

    /// Number of color output images.
    fn colors() -> usize;

    /// Is depth image used.
    fn depth() -> bool;

    /// Format of vertices.
    fn vertices() -> &[(&[Element<Format>], ElemStride)];

    /// Bindings for the pass.
    fn bindings() -> &[DescriptorSetLayoutBinding];
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
    fn load_shader_set(storage: &mut Vec<B::ShaderModule>, device: &mut D, aux: &mut T) -> GraphicsShaderSet<B>;

    /// Build pass instance.
    fn build(
        frames: usize,
        device: &mut D,
        aux: &mut T,
    ) -> Self;

    /// Record drawing commands to the command buffer provided.
    fn run<I>(&mut self, sampled: I, frame: usize, device: &mut D, aux: &T, &mut CommandBuffer<Graphics, B>)
    where
        I: IntoIterator,
        I::Item: Borrow<B::Image>,
    ;

    /// Access image resource storage in `T`.
    fn image(id: ImageId, aux: &mut T) -> Option<&mut I>,
}

struct Resources<B: Backend> {
    framebuffer: B::Framebuffer,
    pool: CommandPool<B, Graphics>,
    individual_reset_pool: CommandPool<B, Graphics>,
}

/// Render pass node.
pub struct RenderPassNode<R> {
    resources: Vec<Resources<B>>,
    render_pass: B::RenderPass,
    pipeline_layout: B::PipelineLayout,
    graphics_pipeline: B::GraphicsPipeline,
    pass: R,
}

/// Overall description for node.
impl<R> NodeDesc for RenderPassNode<R>
where
    R: RenderPassDesc,
{
    type Buffers = Empty<(buffer::Usage, buffer::State, PipelineStage)>;

    type Images = SmallVec<[(image::Usage, image::State, PipelineStage); 16]>;

    type Capability = Graphics;

    fn name() -> &str {
        R::name()
    }

    fn buffers() -> Self::Buffers {
        empty()
    }

    fn images() -> Self::Images {
        let sampled = (0 .. R::sampled()).map(|_| (image::Usage::SAMPLED, (image::Access::SHADER_READ, image::Layout::ShaderReadOnlyOptimal), all_graphics_shaders_stages()));
        let colors = (0 .. R::colors()).map(|_| (image::Usage::COLOR_ATTACHMENT, (image::Access::COLOR_ATTACHMENT_READ | image::Access::COLOR_ATTACHMENT_WRITE, image::Layout::ColorAttachmentOptimal), all_graphics_shaders_stages()));
        let depth = if R::depth() { Some((image::Usage::DEPTH_STENCIL_ATTACHMENT, (image::Access::DEPTH_STENCIL_ATTACHMENT_READ | image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE, DepthStencilAttachmentOptimal), all_graphics_shaders_stages())) } else { None };

        sampled.chain(colors).chain(depth).collect()
    }
}

impl<'a, B, D, T, R> Submittables<'a, B: Backend, D, T> for RenderPassNode<R> {
    type Submittable = EitherSubmit<'a, B, Graphics>;
    type IntoIter = SmallVec<[EitherSubmit<'a, B, Graphics>; 3]>;
}

impl<R> Node<B, D, T> for RenderPassNode<R>
where
    B: Backend,
    D: Device<B>,
    R: RenderPass<B, D, T>;
{
    fn build<F>(
        buffers: Vec<BufferInfo>,
        images: Vec<ImageInfo>,
        frames: usize,
        extent: Extent,
        pools: F,
        device: &mut D,
        aux: &mut T,
    ) -> Self
    where
        F: FnMut(&mut D, CommandPoolCreateFlags) -> CommandPool<B, Self::Capability>,
    {
        trace!("Creating RenderPass instance for '{}'", R::name());
        let render_pass: B::RenderPass = {
            let attachments = (0 .. R::colors()).map(|index| Attachment {
                format: Some(unimplemented!()),
                ops: AttachmentOps {
                    load: AttachmentLoadOp::Load,
                    store: AttachmentStoreOp::Store,
                },
                stencil_ops: AttachmentOps::DONT_CARE,
                layouts: images[index + R::sampled()]
            }).chain( if R::depth() {
                Some(Attachment {
                    format: Some(unimplemented!()),
                    ops: AttachmentOps {
                        load: AttachmentLoadOp::Load,
                        store: AttachmentStoreOp::Store,
                    },
                    stencil_ops: AttachmentOps::DONT_CARE,
                    layouts: images[index + R::sampled() - R::colors()]
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
                preserved: &[],
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
        let pipeline_layout = device.create_pipeline_layout(Some(&descriptor_set_layout), empty());

        let graphics_pipeline = {
            let mut shaders = Vec::new();
            let shader_set = R::load_shader_set(&mut shaders, device, aux);

            let mut vertex_buffers = Vec::new();
            let mut attributes = Vec::new();

            for &(attributes, stride) in self.pass.vertices() {
                push_vertex_desc(attributes, stride, &mut vertex_buffers, &mut attributes);
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
                    alpha_coverage: false,
                    logic_op: None,
                    targets: (0 .. R::colors()).map(|index| ColorBlendDesc(ColorMask::ALL, BlendState::ALPHA)).collect(),
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

        RenderPassNode {
            resources: (0 .. frames).map(|_| Resources {
                framebuffer: device.create_framebuffer(&render_pass, &[], extent).unwrap(),
                pool: pools(device, CommandPoolCreateFlags::empty()),
                individual_reset_pool: pools(device, CommandPoolCreateFlags::RESET_INDIVIDUAL),
            }).collect(),
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            pass: R::build(frames, device, aux),
        }
    }

    fn run<'a>(
        &'a mut self,
        frame: usize,
        device: &mut D,
        aux: &'a T,
    ) -> SmallVec<[EitherSubmit<'a, B, Graphics>; 3]>
    {
        unimplemented!()
    }
}


fn all_graphics_shaders_stages() -> PipelineStage {
    PipelineStage::VERTEX_SHADER |
    PipelineStage::DOMAIN_SHADER |
    PipelineStage::HULL_SHADER |
    PipelineStage::GEOMETRY_SHADER |
    PipelineStage::FRAGMENT_SHADER
}

fn push_vertex_desc<B>(
    attributes: &[Element<Format>],
    stride: ElemStride,
    vertex_buffers: &mut Vec<VertexBufferDesc>,
    attributes: &mut Vec<AttributeDesc>,
) where
    B: Backend,
{
    let index = vertex_buffers.len() as BufferIndex;

    vertex_buffers
        .push(VertexBufferDesc { stride, rate: 0 });

    let mut location = attributes
        .last()
        .map(|a| a.location + 1)
        .unwrap_or(0);
    for &attribute in attributes {
        attributes.push(AttributeDesc {
            location,
            binding: index,
            element: attribute,
        });
        location += 1;
    }
}

fn depth_stencil_desc(depth: bool) -> Option<DepthStencilDesc> {
    
}
