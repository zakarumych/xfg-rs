use std::borrow::Borrow;
use std::error::Error;
use std::fmt;
use std::ops::Range;

use gfx_hal::{Backend, Device};
use gfx_hal::device::{Extent, FramebufferError, ShaderError};
use gfx_hal::format::{AspectFlags, Format, Swizzle};
use gfx_hal::image::{AaMode, Kind, Level, SubresourceRange, Usage};
use gfx_hal::memory::Properties;
use gfx_hal::pso::{CreationError, PipelineStage};
use gfx_hal::window::Backbuffer;

use attachment::{Attachment, AttachmentRef, AttachmentDesc};
use graph::Graph;
use pass::{PassBuilder, PassShaders, PassNode};



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
    InvalidConfiguaration,
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
            GraphBuildError::InvalidConfiguaration => write!(fmt, "Graph has invalid configuration"),
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
    attachments: Vec<Attachment>,
    passes: Vec<PassBuilder<P>>,
    present: Option<AttachmentRef>,
    extent: Extent,
}

impl<P> GraphBuilder<P> {
    /// Create a new `GraphBuilder`
    pub fn new() -> Self {
        GraphBuilder {
            attachments: Vec::new(),
            passes: Vec::new(),
            present: None,
            extent: Extent {
                width: 0,
                height: 0,
                depth: 0,
            },
        }
    }

    /// Add an `Attachment` to the `Graph` and return a value to reference added attachment.
    /// 
    /// ### Parameters:
    /// 
    /// - `attachment`: attachment description.
    /// 
    pub fn add_attachment<A>(&mut self, attachment: A) -> AttachmentRef
    where
        A: Into<Attachment>,
    {
        self.attachments.push(attachment.into());
        AttachmentRef(self.attachments.len() - 1)
    }

    /// Add a few `Attachment`s to the `Graph` and return an iterator of references to added attachment.
    /// 
    /// ### Parameters:
    /// 
    /// - `attachments`: iterator that yields attachment descriptions.
    /// 
    pub fn add_attachments<I, A>(&mut self, attachments: I) -> Range<AttachmentRef>
    where
        I: IntoIterator<Item=A>,
        A: Into<Attachment>,
    {
        let start = self.attachments.len();
        self.attachments.extend(attachments.into_iter().map(Into::into));
        AttachmentRef(start) .. AttachmentRef(self.attachments.len())
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
        A: FnMut(Kind, Level, Format, Usage, Properties, &B::Device) -> Result<I, E>,
        I: Borrow<B::Image>,
        P: PassShaders<B>,
    {
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
                            self.attachments[present.0].format,
                            Swizzle::NO,
                            SubresourceRange {
                                aspects: self.attachments[present.0].format.aspect_flags(),
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

        let mut attachments = self.attachments.into_iter().map(|a| {
            AttachmentDesc {
                format: a.format,
                clear: a.clear,
                write: None,
                read: None,
                images: None,
                views: None,
                is_surface: false,
            }
        }).collect::<Vec<_>>();

        attachments[present.0].views = Some(0 .. image_views.len());
        attachments[present.0].is_surface = true;

        info!("Reorder passes to maximize overlapping");
        // Reorder passes to maximise overlapping
        // while keeping all dependencies before dependants.
        let (passes, deps) = reorder_passes(self.passes);

        let mut images = vec![];

        info!("Build pass nodes from pass builders");
        let mut pass_nodes: Vec<PassNode<B, P>> = Vec::new();

        for (pass_index, pass) in passes.iter().enumerate() {
            info!("Check sampled inputs");
            for &sampled in &pass.sampled {
                let ref mut sampled = attachments[sampled.0];
                debug_assert!(sampled.write.is_some());
                debug_assert!(sampled.views.is_some());
                debug_assert!(sampled.images.is_some());
                sampled.read.get_or_insert_with(|| pass_index .. pass_index).end = pass_index;
            }

            info!("Check sampled targets");
            for &sampled in &pass.sampled {
                let ref mut sampled = attachments[sampled.0];
                debug_assert!(sampled.write.is_some());
                debug_assert!(sampled.views.is_some());
                debug_assert!(sampled.images.is_some());
                sampled.read.get_or_insert_with(|| pass_index .. pass_index).end = pass_index;
            }

            info!("Check input targets");
            for &input in &pass.inputs {
                let ref mut input = attachments[input.0];
                debug_assert!(input.write.is_some());
                debug_assert!(input.views.is_some());
                debug_assert!(input.images.is_some());
                input.read.get_or_insert_with(|| pass_index .. pass_index).end = pass_index;
            }

            info!("Create color targets");
            for &color in &pass.colors {
                let ref mut color = attachments[color.0.index()];
                color.write.get_or_insert_with(|| pass_index .. pass_index).end = pass_index;
                if color.views.is_none() {
                    debug_assert!(color.images.is_none());
                    create_target::<B, _, I, E>(
                        color.format,
                        &mut allocator,
                        device,
                        &mut images,
                        &mut image_views,
                        self.extent,
                        frames,
                    ).map_err(GraphBuildError::AllocationError)?;
                    color.views = Some((image_views.len() - frames .. image_views.len()));
                    color.images = Some((images.len() - frames .. images.len()));
                }
            }

            info!("Create depth-stencil target");
            if let Some(depth_stencil) = pass.depth_stencil {
                let ref mut depth_stencil = attachments[depth_stencil.0.index()];
                depth_stencil.write.get_or_insert_with(|| pass_index .. pass_index).end = pass_index;
                if depth_stencil.views.is_none() {
                    debug_assert!(depth_stencil.images.is_none());
                    create_target::<B, _, I, E>(
                        depth_stencil.format,
                        &mut allocator,
                        device,
                        &mut images,
                        &mut image_views,
                        self.extent,
                        frames,
                    ).map_err(GraphBuildError::AllocationError)?;
                    depth_stencil.views = Some((image_views.len() - frames .. image_views.len()));
                    depth_stencil.images = Some((images.len() - frames .. images.len()));
                }
            }
        }

        for ((pass_index, pass), last_dep) in passes.into_iter().enumerate().zip(deps) {
            let mut node =
                pass.build(device, self.extent, &attachments, &image_views, &images, pass_index)?;

            if let Some(last_dep) = last_dep {
                node.depends = if pass_nodes
                    .iter()
                    .find(|node| {
                        node.depends
                            .as_ref()
                            .map(|&(id, _)| id == last_dep)
                            .unwrap_or(false)
                    })
                    .is_none()
                {
                    // No passes prior this depends on `last_dep`
                    Some((last_dep, PipelineStage::TOP_OF_PIPE)) // Pick better stage.
                } else {
                    None
                };
            }

            pass_nodes.push(node);
        }

        info!("Create semaphores");
        let mut signals = Vec::new();
        for i in 0..pass_nodes.len() {
            if let Some(j) = pass_nodes.iter().position(|node| {
                node.depends
                    .as_ref()
                    .map(|&(id, _)| id == i)
                    .unwrap_or(false)
            }) {
                // j depends on i
                assert!(
                    pass_nodes
                        .iter()
                        .skip(j + 1)
                        .find(|node| node.depends
                            .as_ref()
                            .map(|&(id, _)| id == i)
                            .unwrap_or(false))
                        .is_none()
                );
                signals.push(Some(device.create_semaphore()));
            } else {
                signals.push(None);
            }
        }

        Ok(Graph {
            passes: pass_nodes,
            signals,
            images,
            views: image_views,
            frames,
            draws_to_surface: attachments[present.0].write.clone().unwrap(),
        })
    }
}

fn reorder_passes<P>(
    mut unscheduled: Vec<PassBuilder<P>>,
) -> (Vec<PassBuilder<P>>, Vec<Option<usize>>) {
    // Ordered passes
    let mut scheduled = vec![];
    let mut deps = vec![];

    // Until we schedule all unscheduled passes
    while !unscheduled.is_empty() {
        // Walk over unscheduled
        let (last_dep, index) = (0..unscheduled.len())
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
        deps.push(last_dep);
    }
    (scheduled, deps)
}

/// Get dependencies of pass.
fn direct_dependencies<P>(
    passes: &[PassBuilder<P>],
    pass: &PassBuilder<P>,
) -> Vec<usize> {
    let mut deps = Vec::new();
    for &input in pass.inputs.iter().chain(&pass.sampled) {
        deps.extend(
            passes
                .iter()
                .enumerate()
                .filter(|p| {
                    p.1.depth_stencil.map(|(a, _)| a) == Some(input)
                        || p.1
                            .colors
                            .iter()
                            .any(|&(a, _)| input == a)
                })
                .map(|p| p.0),
        );
    }
    deps.sort();
    deps.dedup();
    deps
}

/// Get other passes that shares output attachments
fn siblings<P>(
    passes: &[PassBuilder<P>],
    pass: &PassBuilder<P>,
) -> Vec<usize> {
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
fn dependencies<P>(
    passes: &[PassBuilder<P>],
    pass: &PassBuilder<P>,
) -> Vec<usize> {
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
    mut allocator: A,
    device: &B::Device,
    images: &mut Vec<I>,
    views: &mut Vec<B::ImageView>,
    extent: Extent,
    frames: usize,
) -> Result<(), E>
where
    B: Backend,
    A: FnMut(Kind, Level, Format, Usage, Properties, &B::Device) -> Result<I, E>,
    I: Borrow<B::Image>,
{
    debug!("Create target with format: {:#?}", format);
    let kind = Kind::D2(extent.width as u16, extent.height as u16, AaMode::Single);
    for _ in 0..frames {
        let image = allocator(
            kind,
            1,
            format,
            if format.aspect_flags().contains(AspectFlags::DEPTH) {
                Usage::DEPTH_STENCIL_ATTACHMENT
            } else {
                Usage::COLOR_ATTACHMENT
            } | Usage::STORAGE,
            Properties::DEVICE_LOCAL,
            device,
        )?;
        let view = device
            .create_image_view(image.borrow(), format, Swizzle::NO, SubresourceRange {
                aspects: format.aspect_flags(),
                layers: 0..1,
                levels: 0..1,
            })
            .expect("Views are expected to be created");
        views.push(view);
        images.push(image);
    }
    Ok(())
}
