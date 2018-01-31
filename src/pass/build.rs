use gfx_hal::{Backend, Device, Primitive};
use gfx_hal::command::{ClearColor, ClearValue};
use gfx_hal::device::Extent;
use gfx_hal::format::Format;
use gfx_hal::image;
use gfx_hal::pass;
use gfx_hal::pso;

use smallvec::SmallVec;

use attachment::{Attachment, AttachmentImageViews, ColorAttachment, ColorAttachmentDesc,
                 DepthStencilAttachment, DepthStencilAttachmentDesc, InputAttachmentDesc};
use descriptors::DescriptorPool;
use frame::SuperFramebuffer;
use graph::GraphBuildError;
use pass::{Pass, PassNode};

/// Collection of data required to construct the node in the rendering `Graph` for a single `Pass`
///
/// ### Type parameters:
///
/// - `B`: hal `Backend`
/// - `T`: auxiliary data used by the inner `Pass`
#[derive(Derivative)]
#[derivative(Debug(bound = ""))]
pub struct PassBuilder<'a, B: Backend, T> {
    pub(crate) inputs: Vec<Option<Attachment<'a>>>,
    pub(crate) colors: Vec<Option<&'a ColorAttachment>>,
    pub(crate) depth_stencil: Option<(Option<&'a DepthStencilAttachment>, bool)>,
    rasterizer: pso::Rasterizer,
    primitive: Primitive,
    pass: Box<Pass<B, T>>,
}

impl<'a, B, T> PassBuilder<'a, B, T>
where
    B: Backend,
{
    /// Construct a `PassBuilder` using the given `Pass`.
    pub fn new<P>(pass: P) -> Self
    where
        P: Pass<B, T> + 'static,
    {
        PassBuilder {
            inputs: vec![None; pass.inputs()],
            colors: vec![None; pass.colors()],
            depth_stencil: if pass.depth() {
                Some((None, pass.stencil()))
            } else {
                None
            },
            rasterizer: pso::Rasterizer::FILL,
            primitive: Primitive::TriangleList,
            pass: Box::new(pass),
        }
    }

    /// Set the input attachment for the given index.
    ///
    /// ### Parameters:
    ///
    /// - `index`: index into the inner pass required input attachments
    /// - `input`: the input attachment to use
    pub fn with_input(mut self, index: usize, input: Attachment<'a>) -> Self {
        self.set_input(index, input);
        self
    }

    /// Set the input attachment for the given index.
    ///
    /// ### Parameters:
    ///
    /// - `index`: index into the inner pass required input attachments
    /// - `input`: the input attachment to use
    pub fn set_input(&mut self, index: usize, input: Attachment<'a>) {
        self.inputs[index] = Some(input);
    }

    /// Set the color attachment for the given index.
    ///
    /// ### Parameters:
    ///
    /// - `index`: index into the inner pass required color attachments
    /// - `color`: the color attachment to use
    pub fn with_color(mut self, index: usize, color: &'a ColorAttachment) -> Self {
        self.set_color(index, color);
        self
    }

    /// Set the color attachment for the given index.
    ///
    /// ### Parameters:
    ///
    /// - `index`: index into the inner pass required color attachments
    /// - `color`: the color attachment to use
    pub fn set_color(&mut self, index: usize, color: &'a ColorAttachment) {
        self.colors[index] = Some(color);
    }

    /// Set the depth stencil attachment to use for the pass.
    ///
    /// Will only be set if the actual `Pass` is configured to use the depth stencil buffer.
    ///
    /// ### Parameters:
    ///
    /// - `depth_stencil`: depth stencil attachment to use
    pub fn with_depth(mut self, depth_stencil: &'a DepthStencilAttachment) -> Self {
        self.set_depth(depth_stencil);
        self
    }

    /// Set the depth stencil attachment to use for the pass.
    ///
    /// Will only be set if the actual `Pass` is configured to use the depth stencil buffer.
    ///
    /// ### Parameters:
    ///
    /// - `depth_stencil`: depth stencil attachment to use
    pub fn set_depth(&mut self, depth_stencil: &'a DepthStencilAttachment) {
        match self.depth_stencil {
            Some((ref mut attachment, _)) => *attachment = Some(depth_stencil),
            None => {}
        }
    }

    /// Build the `PassNode` that will be added to the rendering `Graph`.
    pub(crate) fn build<E>(
        self,
        device: &B::Device,
        inputs: &[InputAttachmentDesc<B>],
        colors: &[ColorAttachmentDesc<B>],
        depth_stencil: Option<DepthStencilAttachmentDesc<B>>,
        extent: Extent,
    ) -> Result<PassNode<B, T>, GraphBuildError<E>> {
        info!("Build pass from {:?}", self);

        // Check attachments
        assert_eq!(inputs.len(), self.pass.inputs());
        assert_eq!(colors.len(), self.pass.colors());
        assert_eq!(
            depth_stencil.is_some(),
            (self.pass.depth() || self.pass.stencil())
        );

        assert!(
            inputs.iter().map(|input| Some(input.format)).eq(self.inputs
                .iter()
                .map(|input| input.map(|input| input.format())))
        );
        assert!(
            colors.iter().map(|color| Some(color.format)).eq(self.colors
                .iter()
                .map(|color| color.map(|color| color.format)))
        );
        assert_eq!(
            depth_stencil.as_ref().map(|depth| Some(depth.format)),
            self.depth_stencil
                .as_ref()
                .map(|depth| depth.0.map(|depth| depth.format))
        );

        // Construct `RenderPass`
        // with single `Subpass` for now
        let renderpass = {
            // Configure input attachments first
            let inputs = inputs.iter().map(|input| {
                let attachment = pass::Attachment {
                    format: Some(input.format),
                    ops: pass::AttachmentOps {
                        load: pass::AttachmentLoadOp::Load,
                        store: pass::AttachmentStoreOp::Store,
                    },
                    stencil_ops: pass::AttachmentOps::DONT_CARE,
                    layouts: image::ImageLayout::General..image::ImageLayout::General,
                };
                debug!("Init input attachment: {:?}", attachment);
                attachment
            });

            // Configure color attachments next to input
            let colors = colors.iter().map(|color| {
                let attachment = pass::Attachment {
                    format: Some(color.format),
                    ops: pass::AttachmentOps {
                        load: if color.clear.is_some() {
                            pass::AttachmentLoadOp::Clear
                        } else {
                            pass::AttachmentLoadOp::Load
                        },
                        store: pass::AttachmentStoreOp::Store,
                    },
                    stencil_ops: pass::AttachmentOps::DONT_CARE,
                    layouts: if color.clear.is_some() {
                        image::ImageLayout::Undefined
                    } else {
                        image::ImageLayout::General
                    }..image::ImageLayout::General,
                };
                debug!("Init color attachment: {:?}", attachment);
                attachment
            });

            // Configure depth-stencil attachments last
            let depth_stencil = depth_stencil.as_ref().map(|depth_stencil| {
                let attachment = pass::Attachment {
                    format: Some(depth_stencil.format),
                    ops: pass::AttachmentOps {
                        load: if self.pass.depth() && depth_stencil.clear.is_some() {
                            pass::AttachmentLoadOp::Clear
                        } else if self.pass.depth() {
                            pass::AttachmentLoadOp::Load
                        } else {
                            pass::AttachmentLoadOp::DontCare
                        },
                        store: if self.pass.depth() {
                            pass::AttachmentStoreOp::Store
                        } else {
                            pass::AttachmentStoreOp::DontCare
                        },
                    },
                    stencil_ops: pass::AttachmentOps {
                        load: if self.pass.stencil() {
                            pass::AttachmentLoadOp::Load
                        } else {
                            pass::AttachmentLoadOp::DontCare
                        },
                        store: if self.pass.stencil() {
                            pass::AttachmentStoreOp::Store
                        } else {
                            pass::AttachmentStoreOp::DontCare
                        },
                    },
                    layouts: if depth_stencil.clear.is_some() {
                        image::ImageLayout::Undefined
                    } else {
                        image::ImageLayout::General
                    }..image::ImageLayout::General,
                };
                debug!("Init depth attachment {:?}", attachment);
                attachment
            });

            let depth_stencil_ref = depth_stencil.as_ref().map(|_| {
                (
                    inputs.len() + colors.len(),
                    image::ImageLayout::DepthStencilAttachmentOptimal,
                )
            });

            // Configure the only `Subpass` using all attachments
            let subpass = pass::SubpassDesc {
                colors: &(0..colors.len())
                    .map(|i| (i + inputs.len(), image::ImageLayout::ColorAttachmentOptimal))
                    .collect::<Vec<_>>(),
                depth_stencil: depth_stencil_ref.as_ref(),
                inputs: &(0..inputs.len())
                    .map(|i| (i, image::ImageLayout::ShaderReadOnlyOptimal))
                    .collect::<Vec<_>>(),
                preserves: &[],
            };

            info!("Create randerpass");
            let renderpass = device.create_render_pass(
                &inputs
                    .chain(colors)
                    .chain(depth_stencil)
                    .collect::<Vec<_>>(),
                &[subpass],
                &[], // TODO: Add external subpass dependency
            );

            debug!("Randerpass: {:?}", renderpass);
            renderpass
        };

        let descriptors = DescriptorPool::new(&self.pass.bindings(), device);

        info!("Create pipeline layout");
        let pipeline_layout = device.create_pipeline_layout(&[descriptors.layout()], &[]);
        debug!("Pipeline layout: {:?}", pipeline_layout);

        let mut shaders = SmallVec::new();
        info!("Create graphics pipeline");
        let graphics_pipeline = {
            // Init basic configuration
            let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
                self.pass.shaders(&mut shaders, device)?,
                self.primitive,
                self.rasterizer.clone(),
                &pipeline_layout,
                pass::Subpass {
                    index: 0,
                    main_pass: &renderpass,
                },
            );

            // Default configuration for blending targets for all color targets
            pipeline_desc.blender.targets =
                vec![
                    pso::ColorBlendDesc(pso::ColorMask::ALL, pso::BlendState::ALPHA);
                    self.pass.colors()
                ];

            // Default configuration for depth-stencil
            pipeline_desc.depth_stencil = Some(pso::DepthStencilDesc {
                depth: pso::DepthTest::On {
                    fun: pso::Comparison::LessEqual,
                    write: true,
                },
                depth_bounds: false,
                stencil: pso::StencilTest::Off,
            });

            // Add all vertex descriptors
            for &(attributes, stride) in self.pass.vertices() {
                push_vertex_desc(attributes, stride, &mut pipeline_desc);
            }

            // Create `GraphicsPipeline`
            let graphics_pipeline = device
                .create_graphics_pipelines(&[pipeline_desc])
                .pop()
                .unwrap()?;
            
            debug!("Graphics pipeline: {:?}", graphics_pipeline);
            graphics_pipeline
        };

        for module in shaders {
            device.destroy_shader_module(module);
        }

        // This color will be set to targets that aren't get cleared
        let ignored_color = ClearColor::Float([0.1, 0.2, 0.3, 1.0]);

        // But we need `ClearValue` for each target
        let mut clears = vec![ClearValue::Color(ignored_color); inputs.len()];

        // Add those for colors
        clears.extend(
            colors
                .iter()
                .map(|c| c.clear.unwrap_or(ignored_color))
                .map(ClearValue::Color),
        );

        // And depth-stencil
        clears.extend(
            depth_stencil
                .as_ref()
                .and_then(|ds| ds.clear)
                .map(ClearValue::DepthStencil),
        );

        debug!("Clear values: {:?}", clears);

        // create framebuffers
        let framebuffer: SuperFramebuffer<B> = {
            if colors.len() == 1 && match colors[0].view {
                AttachmentImageViews::External => true,
                _ => false,
            } {
                SuperFramebuffer::External
            } else {
                let mut frames = None;

                colors
                    .iter()
                    .map(|c| &c.view)
                    .chain(depth_stencil.iter().map(|ds| &ds.view))
                    .map(|view| match *view {
                        AttachmentImageViews::Owned(ref images) => images,
                        AttachmentImageViews::External => {
                            panic!("External framebuffer isn't valid for multicolor output")
                        }
                    })
                    .for_each(|images| {
                        let frames = frames.get_or_insert_with(|| vec![vec![]; images.len()]);
                        assert_eq!(frames.len(), images.len());
                        for i in 0..images.len() {
                            frames[i].push(&images[i]);
                        }
                    });

                let frames = frames.unwrap_or(vec![]);

                // transpose frame matrix
                if frames.len() > 1 {
                    assert!(
                        frames[1..]
                            .iter()
                            .all(|targets| targets.len() == frames[0].len())
                    );
                }

                SuperFramebuffer::Owned(frames
                    .iter()
                    .map(|targets| device.create_framebuffer(&renderpass, targets, extent))
                    .collect::<Result<Vec<_>, _>>()?)
            }
        };

        debug!("Framebuffer: {:?}", framebuffer);

        Ok(PassNode {
            clears,
            descriptors,
            pipeline_layout,
            graphics_pipeline,
            renderpass,
            framebuffer,
            pass: self.pass,
            depends: None,
        })
    }
}

fn push_vertex_desc<B>(
    attributes: &[pso::Element<Format>],
    stride: pso::ElemStride,
    pipeline_desc: &mut pso::GraphicsPipelineDesc<B>,
) where
    B: Backend,
{
    let index = pipeline_desc.vertex_buffers.len() as pso::BufferIndex;

    pipeline_desc
        .vertex_buffers
        .push(pso::VertexBufferDesc { stride, rate: 0 });

    let mut location = pipeline_desc
        .attributes
        .last()
        .map(|a| a.location + 1)
        .unwrap_or(0);
    for &attribute in attributes {
        pipeline_desc.attributes.push(pso::AttributeDesc {
            location,
            binding: index,
            element: attribute,
        });
        location += 1;
    }
}
