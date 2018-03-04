use std::collections::hash_map::{Entry, HashMap};
use std::iter::FromIterator;
use std::ops::Range;

use gfx_hal::{Backend, Device, Primitive};
use gfx_hal::command::{ClearColor, ClearDepthStencil, ClearValue};
use gfx_hal::device::Extent;
use gfx_hal::format::{Aspects, Format};
use gfx_hal::image;
use gfx_hal::pass;
use gfx_hal::pso;

use smallvec::SmallVec;

use attachment::{AttachmentDesc, AttachmentImages, AttachmentRef};
use descriptors::DescriptorPool;
use frame::SuperFramebuffer;
use graph::GraphBuildError;
use pass::{PassDesc, PassNode, PassShaders};
use utils::common_image_layout;

use chain::*;

/// Collection of data required to construct the node in the rendering `Graph` for a single `Pass`
///
/// ### Type parameters:
///
/// - `B`: hal `Backend`
/// - `T`: auxiliary data used by the inner `Pass`
#[derive(Debug)]
pub struct PassBuilder<P> {
    pub(crate) sampled: Vec<AttachmentRef>,
    pub(crate) storages: Vec<AttachmentRef>,
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
            storages: Vec::new(),
            inputs: Vec::new(),
            colors: Vec::new(),
            depth_stencil: None,
            rasterizer: pso::Rasterizer::FILL,
            primitive: Primitive::TriangleList,
            pass,
        }
    }

    /// Specify attachment to be sampled in pass.
    ///
    /// ### Parameters:
    ///
    /// - `input`: attachment to use
    pub fn with_sampled(mut self, input: AttachmentRef) -> Self {
        self.sampled.push(input);
        self
    }

    /// Specify attachment to be sampled in pass.
    ///
    /// ### Parameters:
    ///
    /// - `input`: attachment to use
    pub fn add_sampled(&mut self, input: AttachmentRef) -> &mut Self {
        self.sampled.push(input);
        self
    }

    /// Specify attachment to be read as storage in pass.
    ///
    /// ### Parameters:
    ///
    /// - `input`: attachment to use
    pub fn with_storage(mut self, input: AttachmentRef) -> Self {
        self.storages.push(input);
        self
    }

    /// Specify attachment to be read as storage in pass.
    ///
    /// ### Parameters:
    ///
    /// - `input`: attachment to use
    pub fn add_storage(&mut self, input: AttachmentRef) -> &mut Self {
        self.storages.push(input);
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
    /// - `input`: attachment to use
    pub fn add_input(&mut self, input: AttachmentRef) -> &mut Self {
        self.inputs.push(input);
        self
    }

    /// Add the color attachment.
    ///
    /// ### Parameters:
    ///
    /// - `color`: attachment to use
    /// - `blend`: blending description to use
    pub fn with_color_blend(mut self, color: AttachmentRef, blend: pso::ColorBlendDesc) -> Self {
        self.colors.push((color, blend));
        self
    }

    /// Add the color attachment.
    ///
    /// ### Parameters:
    ///
    /// - `color`: attachment to use
    /// - `blend`: blending description to use
    pub fn add_color_blend(
        &mut self,
        color: AttachmentRef,
        blend: pso::ColorBlendDesc,
    ) -> &mut Self {
        self.colors.push((color, blend));
        self
    }

    /// Add the color attachment.
    ///
    /// ### Parameters:
    ///
    /// - `color`: attachment to use
    pub fn with_color(mut self, color: AttachmentRef) -> Self {
        self.colors.push((color, pso::ColorBlendDesc::EMPTY));
        self
    }

    /// Add the color attachment.
    ///
    /// ### Parameters:
    ///
    /// - `color`: attachment to use
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
    /// - `depth_stencil`: attachment to use
    pub fn with_depth_stencil_desc(
        mut self,
        depth_stencil: AttachmentRef,
        desc: pso::DepthStencilDesc,
    ) -> Self {
        self.depth_stencil = Some((depth_stencil, desc));
        self
    }

    /// Set the depth stencil attachment to use for the pass.
    ///
    /// Will only be set if the actual `Pass` is configured to use the depth stencil buffer.
    ///
    /// ### Parameters:
    ///
    /// - `depth_stencil`: attachment to use
    pub fn set_depth_stencil_desc(
        &mut self,
        depth_stencil: AttachmentRef,
        desc: pso::DepthStencilDesc,
    ) -> &mut Self {
        self.depth_stencil = Some((depth_stencil, desc));
        self
    }

    /// Set the depth stencil attachment to use for the pass.
    ///
    /// Will only be set if the actual `Pass` is configured to use the depth stencil buffer.
    ///
    /// ### Parameters:
    ///
    /// - `depth_stencil`: attachment to use
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
    /// - `depth_stencil`: attachment to use
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

    /// Collect links of the pass.
    /// Chains is built upon them.
    /// This method is invoked with passes already reordered.
    pub(crate) fn links<'a, F>(&self, index: usize, desc: F) -> PassLinks
    where
        F: Fn(AttachmentRef) -> &'a AttachmentDesc,
    {
        let mut attachments: HashMap<AttachmentRef, ImageLink> = HashMap::new();

        let put = |entry: Entry<AttachmentRef, ImageLink>, usage, layout, stage, access| match entry
        {
            Entry::Occupied(occupied) => {
                let ref mut dep = *occupied.into_mut();
                dep.stages |= stage;
                dep.state.usage |= usage;
                dep.state.layout = common_image_layout(layout, dep.state.layout);
                dep.state.access |= access;
            }
            Entry::Vacant(vacant) => {
                vacant.insert(Link {
                    id: ChainId::new(vacant.key().index()),
                    stages: stage,
                    state: ImageState {
                        layout,
                        usage,
                        access,
                    },
                });
            }
        };

        for &a in &self.sampled {
            assert!(desc(a).is_read(index));
            put(
                attachments.entry(a),
                image::Usage::SAMPLED,
                image::ImageLayout::ShaderReadOnlyOptimal,
                pso::PipelineStage::FRAGMENT_SHADER,
                image::Access::SHADER_READ,
            );
        }
        for &a in &self.storages {
            assert!(desc(a).is_read(index));
            put(
                attachments.entry(a),
                image::Usage::STORAGE,
                image::ImageLayout::General,
                pso::PipelineStage::FRAGMENT_SHADER,
                image::Access::SHADER_READ,
            );
        }
        for &a in &self.inputs {
            assert!(desc(a).is_read(index));
            put(
                attachments.entry(a),
                image::Usage::INPUT_ATTACHMENT,
                image::ImageLayout::ShaderReadOnlyOptimal,
                pso::PipelineStage::FRAGMENT_SHADER,
                image::Access::INPUT_ATTACHMENT_READ,
            );
        }
        for &(a, _) in &self.colors {
            assert!(desc(a).is_write(index));
            let access = if desc(a).is_read(index) {
                image::Access::COLOR_ATTACHMENT_READ | image::Access::COLOR_ATTACHMENT_WRITE
            } else {
                image::Access::COLOR_ATTACHMENT_WRITE
            };
            put(
                attachments.entry(a),
                image::Usage::COLOR_ATTACHMENT,
                image::ImageLayout::ColorAttachmentOptimal,
                pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                access,
            );
        }
        for &(a, _) in &self.depth_stencil {
            assert!(desc(a).is_write(index));
            let access = if desc(a).is_read(index) {
                image::Access::COLOR_ATTACHMENT_READ | image::Access::COLOR_ATTACHMENT_WRITE
            } else {
                image::Access::COLOR_ATTACHMENT_WRITE
            };
            put(
                attachments.entry(a),
                image::Usage::DEPTH_STENCIL_ATTACHMENT,
                image::ImageLayout::DepthStencilAttachmentOptimal,
                pso::PipelineStage::EARLY_FRAGMENT_TESTS | pso::PipelineStage::LATE_FRAGMENT_TESTS,
                access,
            );
        }

        PassLinks {
            images: attachments.into_iter().map(|(_, l)| l).collect(),
            buffers: Vec::new(),
        }
    }

    /// Build the `PassNode` that will be added to the rendering `Graph`.
    pub(crate) fn build<'a, B, E, V, I>(
        &self,
        index: usize,
        device: &B::Device,
        extent: Extent,
        chains: &GraphChains,
        views: V,
        images: I,
    ) -> Result<PassNode<B, P>, GraphBuildError<E>>
    where
        B: Backend,
        P: PassShaders<B>,
        V: Fn(AttachmentRef) -> &'a [B::ImageView],
        I: Fn(AttachmentRef) -> &'a [B::Image],
    {
        debug!("Build pass from {:?}", self);

        // Check attachments setup
        assert_eq!(self.sampled.len(), self.pass.sampled());
        assert_eq!(self.inputs.len(), self.pass.inputs());
        assert_eq!(self.colors.len(), self.pass.colors());
        assert_eq!(
            self.depth_stencil.is_some(),
            self.pass.depth() || self.pass.stencil()
        );

        // Construct `RenderPass`
        // with single `Subpass` for now
        let renderpass = {
            // Configure input attachments first
            let inputs = self.inputs.iter().map(|&a| {
                let ref link = chains[a];
                let attachment = pass::Attachment {
                    format: None,
                    ops: pass::AttachmentOps {
                        load: link.load_op(index),
                        store: link.store_op(index),
                    },
                    stencil_ops: pass::AttachmentOps::DONT_CARE,
                    layouts: link.pass_layout_transition(index),
                };
                debug!("Init input attachment: {:?}", attachment);
                attachment
            });

            // Configure color attachments next to input
            let colors = self.colors.iter().map(|&(color, _)| {
                let ref link = chains[color];
                let attachment = pass::Attachment {
                    format: None,
                    ops: pass::AttachmentOps {
                        load: link.load_op(index),
                        store: link.store_op(index),
                    },
                    stencil_ops: pass::AttachmentOps::DONT_CARE,
                    layouts: link.pass_layout_transition(index),
                };
                debug!("Init color attachment: {:?}", attachment);
                attachment
            });

            // Configure depth-stencil attachments last
            let depth_stencil = self.depth_stencil.as_ref().map(|&(depth_stencil, _)| {
                let ref link = chains[depth_stencil];
                let attachment = pass::Attachment {
                    format: None,
                    ops: pass::AttachmentOps {
                        load: link.load_op(index),
                        store: link.store_op(index),
                    },
                    stencil_ops: pass::AttachmentOps::DONT_CARE,
                    layouts: link.pass_layout_transition(index),
                };
                debug!("Init depth attachment {:?}", attachment);
                attachment
            });

            // Configure the only `Subpass` using all attachments
            let subpass = pass::SubpassDesc {
                inputs: &self.inputs
                    .iter()
                    .enumerate()
                    .map(|(i, &input)| (i, chains[input].subpass_layout(index)))
                    .collect::<Vec<_>>(),
                colors: &self.colors
                    .iter()
                    .enumerate()
                    .map(|(i, &(color, _))| {
                        (i + self.inputs.len(), chains[color].subpass_layout(index))
                    })
                    .collect::<Vec<_>>(),
                depth_stencil: self.depth_stencil
                    .as_ref()
                    .map(|&(depth_stencil, _)| {
                        (
                            self.inputs.len() + self.colors.len(),
                            chains[depth_stencil].subpass_layout(index),
                        )
                    })
                    .as_ref(),
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
                (0..self.pass.colors()).map(|i| self.colors[i].1).collect();

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
        // ignored value is pink
        let ignored_color = ClearValue::Color(ClearColor::Float([1.0, 0.0, 0.5, 1.0]));
        let ignored_depth = ClearValue::DepthStencil(ClearDepthStencil(1.0, 0));

        // But we need `ClearValue` for each target
        let mut clears = vec![ignored_color; self.inputs.len()];

        // Add those for colors
        clears.extend(
            self.colors
                .iter()
                .map(|&(c, _)| chains[c].clear_value(index).unwrap_or(ignored_color)),
        );

        // And depth-stencil
        clears.extend(
            self.depth_stencil
                .as_ref()
                .map(|&(ds, _)| chains[ds].clear_value(index).unwrap_or(ignored_depth)),
        );

        debug!("Clear values: {:?}", clears);

        // create framebuffers
        let framebuffer: SuperFramebuffer<B> = {
            if self.inputs.len() == 0 && self.colors.len() == 1
                && views(self.colors[0].0).is_empty()
            {
                SuperFramebuffer::External
            } else {
                debug!(
                    "Create framebuffers from:\ninputs: {:#?}\ncolors: {:#?}\ndepth-stencil: {:#?}",
                    self.inputs, self.colors, self.depth_stencil
                );
                let mut frames = None;

                for views in self.inputs
                    .iter()
                    .chain(self.colors.iter().map(|&(ref a, _)| a))
                    .chain(self.depth_stencil.as_ref().map(|&(ref a, _)| a))
                    .map(|&a| views(a))
                {
                    let frames = frames.get_or_insert_with(|| vec![vec![]; views.len()]);
                    assert_eq!(frames.len(), views.len());

                    for (frame, view) in frames.iter_mut().zip(views) {
                        frame.push(view);
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
                    .map(|targets| {
                        device.create_framebuffer(&renderpass, targets.iter().cloned(), extent)
                    })
                    .collect::<Result<Vec<_>, _>>()?)
            }
        };

        let inputs = {
            let mut frames = None;
            debug!(
                "Collect inputs:\nsampeld: {:#?}\nstorages: {:#?}\nattchment: {:#?}",
                self.sampled, self.storages, self.inputs
            );
            for images in self.sampled
                .iter()
                .chain(&self.storages)
                .chain(&self.inputs)
                .map(|&a| images(a))
            {
                let frames = frames.get_or_insert_with(|| vec![vec![]; images.len()]);
                assert_eq!(frames.len(), images.len());
                for (frame, image) in frames.iter_mut().zip(images) {
                    frame.push(image as *const _);
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
