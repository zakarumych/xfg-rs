
use std::{borrow::Borrow, iter::{empty, Empty}, ops::Range};

use hal::{ Backend, Device, buffer, image, Primitive, 
    command::{CommandBuffer, Primary, OneShot},
    format::Format,
    image::Extent,
    queue::Graphics,
    pass::{Attachment, AttachmentOps, AttachmentLoadOp, AttachmentStoreOp, SubpassDesc, Subpass, SubpassDependency},
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{
        Element, ElemStride, VertexBufferDesc, PipelineStage,
        AttributeDesc, InputAssemblerDesc, PrimitiveRestart,
        ColorBlendDesc, ColorMask, BlendState, DepthTest, DepthStencilDesc,
        StencilTest, Comparison, BakedStates, BasePipeline, Rasterizer,
        PipelineCreationFlags, DescriptorSetLayoutBinding, BufferIndex,
        GraphicsShaderSet, BlendDesc, GraphicsPipelineDesc, ShaderStageFlags,
    },
};

use smallvec::SmallVec;

use id::{BufferId, ImageId};
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
        I::Item: Borrow<B::Image>,
    ;
}

struct Resources<B: Backend> {
    framebuffer: B::Framebuffer,
    pool: CommandPool<B, Graphics>,
    individual_reset_pool: CommandPool<B, Graphics>,
}

/// Render pass node.
pub struct RenderPassNode<B: Backend, R> {
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
                format: Some(unimplemented!()),
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
                    format: Some(unimplemented!()),
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

        RenderPassNode {
            resources: (0 .. frames).map(|_| Resources {
                framebuffer: device.create_framebuffer(&render_pass, &[], unimplemented!()).unwrap(),
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
