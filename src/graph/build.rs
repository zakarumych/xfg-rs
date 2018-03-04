use std::borrow::Borrow;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::ops::{Range, RangeFrom};

use gfx_hal::{Backend, Device};
use gfx_hal::device::{Extent, FramebufferError, ShaderError};
use gfx_hal::format::{Format, Swizzle};
use gfx_hal::image::{AaMode, Kind, Level, SubresourceRange, Usage as ImageUsage, ImageLayout, Access as ImageAccess};
use gfx_hal::memory::Properties;
use gfx_hal::pso::{CreationError, PipelineStage};
use gfx_hal::window::Backbuffer;

use attachment::{Attachment, AttachmentDesc, AttachmentRef, AttachmentImages};
use graph::Graph;
use pass::{PassBuilder, PassNode, PassShaders};
use chain::{ChainId, GraphChains, ImageChain, BufferChain, PassLinks, ImageLink, ImageState};

/// Possible errors during graph building
#[derive(Debug, Clone)]
pub enum GraphBuildError<E> {
    /// Framebuffer could not be created
    FramebufferError,
    /// Shader module creation error
    ShaderError(ShaderError),
    /// If no presentation render target is set
    PresentationAttachmentNotSet,
    /// If no backbuffer is set
    BackbufferNotSet,
    /// Allocation errors as returned by the `allocator` function given to `GraphBuilder::build`
    AllocationError(E),
    /// Graph configuration is invalid.
    InvalidConfiguration,
    /// Any other errors encountered during graph building
    Other,
}

impl<E> From<FramebufferError> for GraphBuildError<E> {
    fn from(_: FramebufferError) -> Self {
        GraphBuildError::FramebufferError
    }
}

impl<E> From<ShaderError> for GraphBuildError<E> {
    fn from(error: ShaderError) -> Self {
        GraphBuildError::ShaderError(error)
    }
}

impl<E> From<CreationError> for GraphBuildError<E> {
    fn from(error: CreationError) -> Self {
        match error {
            CreationError::Other => GraphBuildError::Other,
            CreationError::InvalidSubpass(_) => unreachable!("This should never happend"),
            CreationError::Shader(error) => error.into(),
        }
    }
}

impl<A> fmt::Display for GraphBuildError<A>
where
    A: fmt::Display,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str("Failed to build graph: ")?;
        match *self {
            GraphBuildError::FramebufferError => fmt.write_str("framebuffer can't be created"),
            GraphBuildError::ShaderError(ref error) => write!(fmt, "{:?}", error),
            GraphBuildError::PresentationAttachmentNotSet => {
                fmt.write_str("Presentation attachment wasn't set in GraphBuilder")
            }
            GraphBuildError::BackbufferNotSet => {
                fmt.write_str("Presentation attachment wasn't set in GraphBuilder")
            }
            GraphBuildError::AllocationError(ref error) => write!(fmt, "{}", error),
            GraphBuildError::InvalidConfiguration => write!(fmt, "Graph has invalid configuration"),
            GraphBuildError::Other => fmt.write_str("Unknown error has occured"),
        }
    }
}

impl<A> Error for GraphBuildError<A>
where
    A: Error,
{
    fn description(&self) -> &str {
        "Failed to build graph"
    }

    fn cause(&self) -> Option<&Error> {
        match *self {
            GraphBuildError::AllocationError(ref error) => Some(error),
            _ => None,
        }
    }
}

/// Graph builder
///
/// ### Type parameters:
///
/// - `B`: render `Backend`
/// - `T`: auxiliary data used by the `Pass`es in the `Graph`
#[derive(Debug)]
pub struct GraphBuilder<P> {
    attachments: HashMap<AttachmentRef, Attachment>,
    passes: Vec<PassBuilder<P>>,
    present: Option<AttachmentRef>,
    extent: Extent,
    chain_id_image: RangeFrom<usize>,
    chain_id_buffer: RangeFrom<usize>,
}

impl<P> GraphBuilder<P> {
    /// Create a new `GraphBuilder`
    pub fn new() -> Self {
        GraphBuilder {
            attachments: HashMap::new(),
            passes: Vec::new(),
            present: None,
            extent: Extent {
                width: 0,
                height: 0,
                depth: 0,
            },
            chain_id_image: 0 ..,
            chain_id_buffer: 0 ..,
        }
    }

    /// Add an `Attachment` to the `Graph` and return `AttachmentRef` of added attachment.
    ///
    /// ### Parameters:
    ///
    /// - `attachment`: attachment description.
    ///
    pub fn add_attachment<A>(&mut self, attachment: A) -> AttachmentRef
    where
        A: Into<Attachment>,
    {
        self.add_attachments(Some(attachment)).start
    }

    /// Add a few `Attachment`s to the `Graph` and return an iterator over `AttachmentRef` of added attachment.
    ///
    /// ### Parameters:
    ///
    /// - `attachments`: iterator that yields attachment descriptions.
    ///
    pub fn add_attachments<I, A>(&mut self, attachments: I) -> Range<AttachmentRef>
    where
        I: IntoIterator<Item = A>,
        A: Into<Attachment>,
    {
        let start = self.chain_id_image.start;
        self.attachments
            .extend(self.chain_id_image.by_ref().map(AttachmentRef::new).zip(attachments.into_iter().map(Into::into)));
        AttachmentRef::new(start) .. AttachmentRef::new(self.chain_id_image.start)
    }

    /// Add a `Pass` to the `Graph`
    ///
    /// ### Parameters:
    ///
    /// - `pass`: pass builder
    pub fn with_pass(mut self, pass: PassBuilder<P>) -> Self {
        self.add_pass(pass);
        self
    }

    /// Add a `Pass` to the `Graph`
    ///
    /// ### Parameters:
    ///
    /// - `pass`: pass builder
    pub fn add_pass(&mut self, pass: PassBuilder<P>) -> &mut Self {
        self.passes.push(pass);
        self
    }

    /// Set the extent of the framebuffers
    ///
    /// ### Parameters:
    ///
    /// - `extent`: hal `Extent`
    pub fn with_extent(mut self, extent: Extent) -> Self {
        self.set_extent(extent);
        self
    }

    /// Set the extent of the framebuffers
    ///
    /// ### Parameters:
    ///
    /// - `extent`: hal `Extent`
    pub fn set_extent(&mut self, extent: Extent) -> &mut Self {
        self.extent = extent;
        self
    }

    /// Set presentation draw surface
    ///
    /// ### Parameters:
    ///
    /// - `present`: color attachment to use as presentation draw surface for the `Graph`
    pub fn with_present(mut self, present: AttachmentRef) -> Self {
        self.set_present(present);
        self
    }

    /// Set presentation draw surface
    ///
    /// ### Parameters:
    ///
    /// - `present`: color attachment to use as presentation draw surface for the `Graph`
    pub fn set_present(&mut self, present: AttachmentRef) -> &mut Self {
        self.present = Some(present);
        self
    }

    /// Build rendering graph
    ///
    /// ### Parameters:
    ///
    /// - `device`: graphics device
    /// - `allocator`: allocator function used for creating render targets
    ///
    /// ### Type parameters:
    ///
    /// - `A`: allocator function
    /// - `I`: render target image type
    /// - `E`: errors returned by the allocator function
    pub fn build<B, A, I, E>(
        self,
        device: &B::Device,
        backbuffer: &Backbuffer<B>,
        mut allocator: A,
    ) -> Result<Graph<B, I, P>, GraphBuildError<E>>
    where
        B: Backend,
        A: FnMut(Kind, Level, Format, ImageUsage, Properties, &B::Device) -> Result<I, E>,
        I: Borrow<B::Image>,
        P: PassShaders<B>,
    {
        self.passes = reorder_passes(self.passes);

        info!("Building graph from {:?}", self);
        let present = self.present
            .ok_or(GraphBuildError::PresentationAttachmentNotSet)?;

        info!("Collect views from backbuffer");
        // Create views for backbuffer
        let (mut image_views, frames) = match *backbuffer {
            Backbuffer::Images(ref images) => (
                images
                    .iter()
                    .map(|image| {
                        device.create_image_view(
                            image,
                            self.attachments[&present].format,
                            Swizzle::NO,
                            SubresourceRange {
                                aspects: self.attachments[&present].format.aspects(),
                                layers: 0..1,
                                levels: 0..1,
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .expect("Views are epxected to be created"),
                images.len(),
            ),
            Backbuffer::Framebuffer(_) => (vec![], 1),
        };

        let mut descs = self.attachments
            .into_iter()
            .map(|(k, v)| (k, AttachmentDesc {
                format: v.format,
                init: v.init,
                writes: None,
                reads: None,
            }))
            .collect::<HashMap<_, _>>();

        for (pass_index, pass) in self.passes.iter().enumerate() {
            info!("Check sampled targets");
            for sampled in &pass.sampled {
                let ref mut sampled = descs[sampled];
                debug_assert!(sampled.writes.is_some());
                sampled
                    .reads
                    .get_or_insert_with(|| (pass_index, pass_index))
                    .1 = pass_index;
            }

            info!("Check storage targets");
            for storage in &pass.storages {
                let ref mut storage = descs[storage];
                debug_assert!(storage.writes.is_some());
                storage
                    .reads
                    .get_or_insert_with(|| (pass_index, pass_index))
                    .1 = pass_index;
            }

            info!("Check input targets");
            for input in &pass.inputs {
                let ref mut input = descs[input];
                debug_assert!(input.writes.is_some());
                input.reads.get_or_insert_with(|| (pass_index, pass_index)).1 = pass_index;
                unimplemented!()
            }

            info!("Check color targets");
            for &(ref color, _) in &pass.colors {
                let ref mut color = descs[color];
                color
                    .writes
                    .get_or_insert_with(|| (pass_index, pass_index))
                    .1 = pass_index;
            }

            info!("Check depth-stencil target");
            if let Some((ref depth_stencil, _)) = pass.depth_stencil {
                let ref mut depth_stencil = descs[depth_stencil];
                depth_stencil
                    .writes
                    .get_or_insert_with(|| (pass_index, pass_index))
                    .1 = pass_index;
            }
        }

        let mut links: Vec<PassLinks> = self.passes.iter().enumerate().map(|(pass_index, pass)| pass.links(pass_index, |a| &descs[&a])).collect();

        // Add `Presentation` as fake pass.
        links.push(PassLinks {
            buffers: Vec::new(),
            images: vec![
                ImageLink {
                    id: present,
                    stages: PipelineStage::empty(),
                    state: ImageState {
                        usage: ImageUsage::empty(),
                        access: ImageAccess::empty(),
                        layout: ImageLayout::Present,
                    }
                }
            ],
        });

        let mut chains = GraphChains::new(self.chain_id_buffer.start, self.chain_id_image.start, &links);
        for (&attachment, desc) in &descs {
            chains[attachment].init = desc.init;
        }

        let mut images = Vec::new();

        let attachment_images = descs.iter().map(|(&attachment, desc)| {
            let (images, views) = create_target::<B, _, I, E>(
                desc.format,
                chains[attachment].usage,
                &mut allocator,
                device,
                &mut images,
                &mut image_views,
                self.extent,
                frames,
            ).map_err(GraphBuildError::AllocationError)?;
            Ok((attachment, AttachmentImages {
                images,
                views
            }))
        }).collect::<Result<HashMap<_, _>, _>>()?;

        let mut passes = Vec::new();
        info!("Build pass nodes from pass builders");
        for (pass_index, pass) in self.passes.iter().enumerate() {
            let views = |a| &image_views[attachment_images[&a].views];
            let images = |a| &images[attachment_images[&a].images];
            let mut node = pass.build(pass_index, device, self.extent, &chains, views, images)?;
            passes.push(node);
        }

        Ok(Graph {
            passes,
            signals: vec![],
            images,
            views: image_views,
            frames,
        })
    }
}

fn reorder_passes<P>(
    mut unscheduled: Vec<PassBuilder<P>>,
) -> Vec<PassBuilder<P>> {
    // Ordered passes
    let mut scheduled = vec![];
    let mut deps = vec![];

    // Until we schedule all unscheduled passes
    while !unscheduled.is_empty() {
        // Walk over unscheduled
        let (_, index) = (0..unscheduled.len())
            .filter(|&index| {
                // Check if all dependencies are scheduled
                dependencies(&unscheduled, &unscheduled[index]).is_empty()
            }).map(|index| {
                // Find indices for all direct dependencies of the pass
                let dependencies = direct_dependencies(&scheduled, &unscheduled[index]);
                let siblings = siblings(&scheduled, &unscheduled[index]);
                (dependencies.into_iter().chain(siblings).max(), index)
            })
            // Smallest index of last dependency wins. `None < Some(0)`
            .min_by_key(|&(last_dep, _)| last_dep)
            // At least one pass with all dependencies scheduled must be found.
            // Or there is dependency circle in unscheduled left.
            .expect("Circular dependency encountered");

        // Store
        scheduled.push(unscheduled.swap_remove(index));
    }
    scheduled
}

/// Get dependencies of pass.
fn direct_dependencies<P>(passes: &[PassBuilder<P>], pass: &PassBuilder<P>) -> Vec<usize> {
    let mut deps = Vec::new();
    for &input in pass.sampled
        .iter()
        .chain(&pass.storages)
        .chain(&pass.inputs)
    {
        deps.extend(
            passes
                .iter()
                .enumerate()
                .filter(|p| {
                    p.1.depth_stencil.map(|(a, _)| a) == Some(input)
                        || p.1.colors.iter().any(|&(a, _)| input == a)
                })
                .map(|p| p.0),
        );
    }
    deps.sort();
    deps.dedup();
    deps
}

/// Get other passes that shares output attachments
fn siblings<P>(passes: &[PassBuilder<P>], pass: &PassBuilder<P>) -> Vec<usize> {
    let mut siblings = Vec::new();
    for &color in pass.colors.iter() {
        siblings.extend(
            passes
                .iter()
                .enumerate()
                .filter(|p| p.1.colors.iter().any(|&a| a == color))
                .map(|p| p.0),
        );
    }
    if let Some(depth_stencil) = pass.depth_stencil {
        siblings.extend(
            passes
                .iter()
                .enumerate()
                .filter(|p| {
                    p.1
                        .depth_stencil
                        .as_ref()
                        .map_or(false, |&a| a == depth_stencil)
                })
                .map(|p| p.0),
        );
    }
    siblings.sort();
    siblings.dedup();
    siblings
}

/// Get dependencies of pass. And dependencies of dependencies.
fn dependencies<P>(passes: &[PassBuilder<P>], pass: &PassBuilder<P>) -> Vec<usize> {
    let mut deps = direct_dependencies(passes, pass);
    deps = deps.into_iter()
        .flat_map(|dep| dependencies(passes, &passes[dep]))
        .collect();
    deps.sort();
    deps.dedup();
    deps
}

fn create_target<B, A, I, E>(
    format: Format,
    usage: ImageUsage,
    mut allocator: A,
    device: &B::Device,
    images: &mut Vec<I>,
    views: &mut Vec<B::ImageView>,
    extent: Extent,
    frames: usize,
) -> Result<(Range<usize>, Range<usize>), E>
where
    B: Backend,
    A: FnMut(Kind, Level, Format, ImageUsage, Properties, &B::Device) -> Result<I, E>,
    I: Borrow<B::Image>,
{
    debug!(
        "Create target with format: {:#?} and usage: {:#?}",
        format, usage
    );
    let istart = images.len();
    let vstart = views.len();
    let kind = Kind::D2(extent.width as u16, extent.height as u16, AaMode::Single);
    for _ in 0..frames {
        let image = allocator(kind, 1, format, usage, Properties::DEVICE_LOCAL, device)?;
        let view = device
            .create_image_view(
                image.borrow(),
                format,
                Swizzle::NO,
                SubresourceRange {
                    aspects: format.aspects(),
                    layers: 0..1,
                    levels: 0..1,
                },
            )
            .expect("Views are expected to be created");
        views.push(view);
        images.push(image);
    }
    Ok((istart .. images.len(), vstart .. views.len()))
}
