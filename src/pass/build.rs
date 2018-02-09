use std::borrow::Borrow;

use gfx_hal::{Backend, Device, Primitive};
use gfx_hal::command::{ClearColor, ClearValue, ClearDepthStencil};
use gfx_hal::device::Extent;
use gfx_hal::format::Format;
use gfx_hal::image;
use gfx_hal::pass;
use gfx_hal::pso;

use smallvec::SmallVec;

use attachment::{AttachmentRef, AttachmentDesc};
use descriptors::DescriptorPool;
use frame::SuperFramebuffer;
use graph::GraphBuildError;
use pass::{PassShaders, PassDesc, PassNode};

/// Collection of data required to construct the node in the rendering `Graph` for a single `Pass`
///
/// ### Type parameters:
///
/// - `B`: hal `Backend`
/// - `T`: auxiliary data used by the inner `Pass`
#[derive(Debug)]
pub struct PassBuilder<P> {
    pub(crate) sampled: Vec<AttachmentRef>,
    pub(crate) inputs: Vec<AttachmentRef>,
    pub(crate) colors: Vec<(AttachmentRef, pso::ColorBlendDesc)>,
    pub(crate) depth_stencil: Option<(AttachmentRef, pso::DepthStencilDesc)>,
    rasterizer: pso::Rasterizer,
    primitive: Primitive,
    pass: P,
}

impl<P> PassBuilder<P>
where
    P: PassDesc,
{
    /// Construct a `PassBuilder` using the given `Pass`.
    pub fn new(pass: P) -> Self {
        PassBuilder {
            sampled: Vec::new(),
            inputs: Vec::new(),
            colors: Vec::new(),
            depth_stencil: None,
            rasterizer: pso::Rasterizer::FILL,
            primitive: Primitive::TriangleList,
            pass,
        }
    }

    /// Add the sampled attachment.
    ///
    /// ### Parameters:
    ///
    /// - `input`: the input attachment to use
    pub fn with_sampled(mut self, input: AttachmentRef) -> Self {
        self.sampled.push(input);
        self
    }

    /// Add the sampled attachment.
    ///
    /// ### Parameters:
    ///
    /// - `input`: the input attachment to use
    pub fn add_sampled(&mut self, input: AttachmentRef) -> &mut Self {
        self.sampled.push(input);
        self
    }

    /// Add the input attachment.
    ///
    /// ### Parameters:
    ///
    /// - `input`: the input attachment to use
    pub fn with_input(mut self, input: AttachmentRef) -> Self {
        self.inputs.push(input);
        self
    }

    /// Add the input attachment.
    ///
    /// ### Parameters:
    ///
    /// - `input`: the input attachment to use
    pub fn add_input(&mut self, input: AttachmentRef) -> &mut Self {
        self.inputs.push(input);
        self
    }

    /// Add the color attachment.
    ///
    /// ### Parameters:
    ///
    /// - `color`: the color attachment to use
    /// - `blend`: blending description to use
    pub fn with_color_blend(mut self, color: AttachmentRef, blend: pso::ColorBlendDesc) -> Self {
        self.colors.push((color, blend));
        self
    }

    /// Add the color attachment.
    ///
    /// ### Parameters:
    ///
    /// - `color`: the color attachment to use
    /// - `blend`: blending description to use
    pub fn add_color_blend(&mut self, color: AttachmentRef, blend: pso::ColorBlendDesc) -> &mut Self {
        self.colors.push((color, blend));
        self
    }

    /// Add the color attachment.
    ///
    /// ### Parameters:
    ///
    /// - `color`: the color attachment to use
    pub fn with_color(mut self, color: AttachmentRef) -> Self {
        self.colors.push((color, pso::ColorBlendDesc::EMPTY));
        self
    }

    /// Add the color attachment.
    ///
    /// ### Parameters:
    ///
    /// - `color`: the color attachment to use
    pub fn add_color(&mut self, color: AttachmentRef) -> &mut Self {
        self.colors.push((color, pso::ColorBlendDesc::EMPTY));
        self
    }

    /// Set the depth stencil attachment to use for the pass.
    ///
    /// Will only be set if the actual `Pass` is configured to use the depth stencil buffer.
    ///
    /// ### Parameters:
    ///
    /// - `depth_stencil`: depth stencil attachment to use
    pub fn with_depth_stencil_desc(mut self, depth_stencil: AttachmentRef, desc: pso::DepthStencilDesc) -> Self {
        self.depth_stencil = Some((depth_stencil, desc));
        self
    }

    /// Set the depth stencil attachment to use for the pass.
    ///
    /// Will only be set if the actual `Pass` is configured to use the depth stencil buffer.
    ///
    /// ### Parameters:
    ///
    /// - `depth_stencil`: depth stencil attachment to use
    pub fn set_depth_stencil_desc(&mut self, depth_stencil: AttachmentRef, desc: pso::DepthStencilDesc) -> &mut Self {
        self.depth_stencil = Some((depth_stencil, desc));
        self
    }

    /// Set the depth stencil attachment to use for the pass.
    ///
    /// Will only be set if the actual `Pass` is configured to use the depth stencil buffer.
    ///
    /// ### Parameters:
    ///
    /// - `depth_stencil`: depth stencil attachment to use
    pub fn with_depth_stencil(mut self, depth_stencil: AttachmentRef) -> Self {
        self.depth_stencil = Some((depth_stencil, depth_stencil_desc(&self.pass)));
        self
    }

    /// Set the depth stencil attachment to use for the pass.
    ///
    /// Will only be set if the actual `Pass` is configured to use the depth stencil buffer.
    ///
    /// ### Parameters:
    ///
    /// - `depth_stencil`: depth stencil attachment to use
    pub fn set_depth_stencil(&mut self, depth_stencil: AttachmentRef) -> &mut Self {
        self.depth_stencil = Some((depth_stencil, depth_stencil_desc(&self.pass)));
        self
    }

    /// Get name of the `Pass`.
    pub fn name(&self) -> &str
    where
        P: PassDesc,
    {
        self.pass.name()
    }

    /// Build the `PassNode` that will be added to the rendering `Graph`.
    pub(crate) fn build<B, E, I>(
        self,
        device: &B::Device,
        extent: Extent,
        attachments: &[AttachmentDesc],
        views: &[B::ImageView],
        images: &[I],
        index: usize,
    ) -> Result<PassNode<B, P>, GraphBuildError<E>>
    where
        B: Backend,
        P: PassShaders<B>,
        I: Borrow<B::Image>,
    {
        debug!("Build pass from {:?}", self);

        // Check attachments setup
        assert_eq!(self.sampled.len(), self.pass.sampled());
        assert_eq!(self.inputs.len(), self.pass.inputs());
        assert_eq!(self.colors.len(), self.pass.colors());
        assert_eq!(self.depth_stencil.is_some(), self.pass.depth() || self.pass.stencil());

        // Construct `RenderPass`
        // with single `Subpass` for now
        let renderpass = {
            // Configure input attachments first
            let inputs = self.inputs.iter().map(|input| {
                let ref input = attachments[input.0];
                let attachment = pass::Attachment {
                    format: Some(input.format),
                    ops: pass::AttachmentOps {
                        load: input.load_op(index),
                        store: input.store_op(index),
                    },
                    stencil_ops: pass::AttachmentOps::DONT_CARE,
                    layouts: input.image_layout_transition(index),
                };
                debug!("Init input attachment: {:?}", attachment);
                attachment
            });

            // Configure color attachments next to input
            let colors = self.colors.iter().map(|color| {
                let ref color = attachments[color.0.index()];
                let attachment = pass::Attachment {
                    format: Some(color.format),
                    ops: pass::AttachmentOps {
                        load: color.load_op(index),
                        store: color.store_op(index),
                    },
                    stencil_ops: pass::AttachmentOps::DONT_CARE,
                    layouts: color.image_layout_transition(index),
                };
                debug!("Init color attachment: {:?}", attachment);
                attachment
            });

            // Configure depth-stencil attachments last
            let depth_stencil = self.depth_stencil.map(|depth_stencil| {
                let ref depth_stencil = attachments[depth_stencil.0.index()];
                let attachment = pass::Attachment {
                    format: Some(depth_stencil.format),
                    ops: pass::AttachmentOps {
                        load: depth_stencil.load_op(index),
                        store: depth_stencil.store_op(index),
                    },
                    stencil_ops: pass::AttachmentOps::DONT_CARE,
                    layouts: depth_stencil.image_layout_transition(index),
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

            let renderpass = device.create_render_pass(
                &inputs
                    .chain(colors)
                    .chain(depth_stencil)
                    .collect::<Vec<_>>(),
                &[subpass],
                &[],
            );

            debug!("Randerpass: {:?}", renderpass);
            renderpass
        };

        let descriptors = DescriptorPool::new(&self.pass.bindings(), device);

        let pipeline_layout = device.create_pipeline_layout(Some(descriptors.layout()), &[]);
        debug!("Pipeline layout: {:?}", pipeline_layout);

        let mut shaders = SmallVec::new();
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
                (0 .. self.pass.colors()).map(|i|
                    self.colors[i].1
                ).collect();

            // Default configuration for depth-stencil
            pipeline_desc.depth_stencil = self.depth_stencil.map(|(_, desc)| desc);

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
        let ignored_color = ClearValue::Color(ClearColor::Float([0.1, 0.2, 0.3, 1.0]));
        let ignored_depth = ClearValue::DepthStencil(ClearDepthStencil(1.0, 0));

        // But we need `ClearValue` for each target
        let mut clears = vec![ignored_color; self.inputs.len()];

        // Add those for colors
        clears.extend(
            self.colors
                .iter()
                .map(|c| attachments[c.0.index()].clear.unwrap_or(ignored_color))
        );

        // And depth-stencil
        clears.extend(
            self.depth_stencil
                .as_ref()
                .map(|ds| attachments[ds.0.index()].clear.unwrap_or(ignored_depth))
        );

        debug!("Clear values: {:?}", clears);

        // create framebuffers
        let framebuffer: SuperFramebuffer<B> = {
            if self.inputs.len() == 0 && self.colors.len() == 1 && attachments[self.colors[0].0.index()].views == Some(0..0) {
                SuperFramebuffer::External
            } else {
                debug!("Create framebuffers from:\ninputs: {:#?}\ncolors: {:#?}\ndepth-stencil: {:#?}", self.inputs, self.colors, self.depth_stencil);
                let mut frames = None;

                for indices in self.inputs.iter().chain(self.colors.iter().map(|&(ref a, _)|a)).chain(self.depth_stencil.as_ref().map(|&(ref a, _)|a)).map(|a| attachments[a.index()].views.clone()) {
                    let indices = indices.ok_or(GraphBuildError::InvalidConfiguaration)?;
                    let frames = frames.get_or_insert_with(|| vec![vec![]; indices.len()]);
                    assert_eq!(frames.len(), indices.len());
                    for (i, image) in indices.enumerate() {
                        frames[i].push(&views[image]);
                    }
                }

                // Check all frames are same sized.
                let frames = frames.unwrap_or(vec![]);
                if frames.len() > 1 {
                    assert!(
                        frames[1..]
                            .iter()
                            .all(|targets| targets.len() == frames[0].len())
                    );
                }

                SuperFramebuffer::Owned(frames
                    .iter()
                    .map(|targets| device.create_framebuffer(&renderpass, targets.iter().cloned(), extent))
                    .collect::<Result<Vec<_>, _>>()?)
            }
        };

        let inputs = {
            let mut frames = None;
            debug!("Collect inputs:\nsampeld: {:#?}\nattchment: {:#?}", self.sampled, self.inputs);
            for indices in self.sampled.into_iter().chain(self.inputs).map(|a| attachments[a.0].images.clone()) {
                let indices = indices.ok_or(GraphBuildError::InvalidConfiguaration)?;
                let frames = frames.get_or_insert_with(|| vec![vec![]; indices.len()]);
                assert_eq!(frames.len(), indices.len());
                for (i, index) in indices.enumerate() {
                    frames[i].push(images[index].borrow() as *const _);
                }
            }
            frames.unwrap_or(vec![])
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
            inputs,
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


fn depth_stencil_desc<P>(pass: &P) -> pso::DepthStencilDesc
where
    P: PassDesc,
{
    pso::DepthStencilDesc {
        depth: if pass.depth() {
            pso::DepthTest::On {
                fun: pso::Comparison::LessEqual,
                write: true,
            }
        } else {
            pso::DepthTest::Off
        },
        depth_bounds: false,
        stencil: pso::StencilTest::Off,
    }
}
