use std::borrow::Borrow;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::ops::Range;
use std::ptr::eq;

use gfx_hal::{Backend, Device};
use gfx_hal::device::{Extent, FramebufferError, ShaderError};
use gfx_hal::format::{AspectFlags, Format, Swizzle};
use gfx_hal::image::{AaMode, Kind, Level, SubresourceRange, Usage};
use gfx_hal::memory::Properties;
use gfx_hal::pso::{CreationError, PipelineStage};
use gfx_hal::window::Backbuffer;

use attachment::{Attachment, AttachmentImageViews, ColorAttachment, ColorAttachmentDesc,
                 DepthStencilAttachment, DepthStencilAttachmentDesc, InputAttachmentDesc};
use graph::Graph;
use pass::{PassBuilder, PassNode};

/// Color range for render targets
pub const COLOR_RANGE: SubresourceRange = SubresourceRange {
    aspects: AspectFlags::COLOR,
    levels: 0..1,
    layers: 0..1,
};

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
#[derive(Derivative)]
#[derivative(Debug(bound = ""))]
pub struct GraphBuilder<'a, B: Backend, T> {
    passes: Vec<PassBuilder<'a, B, T>>,
    present: Option<&'a ColorAttachment>,
    backbuffer: Option<&'a Backbuffer<B>>,
    extent: Extent,
}

impl<'a, B, T> GraphBuilder<'a, B, T>
where
    B: Backend,
{
    /// Create a new `GraphBuilder`
    pub fn new() -> Self {
        GraphBuilder {
            passes: Vec::new(),
            present: None,
            backbuffer: None,
            extent: Extent {
                width: 0,
                height: 0,
                depth: 0,
            },
        }
    }

    /// Add a `Pass` to the `Graph`
    ///
    /// ### Parameters:
    ///
    /// - `pass`: pass builder
    pub fn with_pass(mut self, pass: PassBuilder<'a, B, T>) -> Self {
        self.add_pass(pass);
        self
    }

    /// Add a `Pass` to the `Graph`
    ///
    /// ### Parameters:
    ///
    /// - `pass`: pass builder
    pub fn add_pass(&mut self, pass: PassBuilder<'a, B, T>) {
        self.passes.push(pass);
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
    pub fn set_extent(&mut self, extent: Extent) {
        self.extent = extent;
    }

    /// Set the backbuffer to use
    ///
    /// ### Parameters:
    ///
    /// - `backbuffer`: hal `Backbuffer`
    pub fn with_backbuffer(mut self, backbuffer: &'a Backbuffer<B>) -> Self {
        self.set_backbuffer(backbuffer);
        self
    }

    /// Set the backbuffer to use
    ///
    /// ### Parameters:
    ///
    /// - `backbuffer`: hal `Backbuffer`
    pub fn set_backbuffer(&mut self, backbuffer: &'a Backbuffer<B>) {
        self.backbuffer = Some(backbuffer);
    }

    /// Set presentation draw surface
    ///
    /// ### Parameters:
    ///
    /// - `present`: color attachment to use as presentation draw surface for the `Graph`
    pub fn with_present(mut self, present: &'a ColorAttachment) -> Self {
        self.set_present(present);
        self
    }

    /// Set presentation draw surface
    ///
    /// ### Parameters:
    ///
    /// - `present`: color attachment to use as presentation draw surface for the `Graph`
    pub fn set_present(&mut self, present: &'a ColorAttachment) {
        self.present = Some(present);
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
    pub fn build<A, I, E>(
        self,
        device: &B::Device,
        mut allocator: A,
    ) -> Result<Graph<B, I, T>, GraphBuildError<E>>
    where
        A: FnMut(Kind, Level, Format, Usage, Properties, &B::Device) -> Result<I, E>,
        I: Borrow<B::Image>,
    {
        info!("Building graph from {:?}", self);
        let present = self.present
            .ok_or(GraphBuildError::PresentationAttachmentNotSet)?;
        let backbuffer = self.backbuffer.ok_or(GraphBuildError::BackbufferNotSet)?;

        info!("Collect views from backbuffer");
        // Create views for backbuffer
        let (mut image_views, frames) = match *backbuffer {
            Backbuffer::Images(ref images) => (
                images
                    .iter()
                    .map(|image| {
                        device.create_image_view(
                            image,
                            present.format,
                            Swizzle::NO,
                            COLOR_RANGE.clone(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .expect("Views are epxected to be created"),
                images.len(),
            ),
            Backbuffer::Framebuffer(_) => (vec![], 1),
        };

        info!("Reorder passes to maximize overlapping");
        // Reorder passes to maximise overlapping
        // while keeping all dependencies before dependants.
        let (passes, deps) = reorder_passes(self.passes);

        info!("Collect all used attachments");
        let color_attachments = color_attachments(&passes);
        let depth_stencil_attachments = depth_stencil_attachments(&passes);

        let mut images = vec![];

        info!("Initialize color targets");
        let mut color_targets = HashMap::<*const _, (Range<usize>, usize)>::new();
        color_targets.insert(present, (0..image_views.len(), 0));
        for attachment in color_attachments {
            if !eq(attachment, present) {
                color_targets.insert(
                    attachment,
                    (
                        create_target::<B, _, I, E>(
                            attachment.format,
                            &mut allocator,
                            device,
                            &mut images,
                            &mut image_views,
                            self.extent,
                            frames,
                            false,
                        ).map_err(GraphBuildError::AllocationError)?,
                        0,
                    ),
                );
            }
        }

        info!("Initialize depth-stencil targets");
        let mut depth_stencil_targets = HashMap::<*const _, (Range<usize>, usize)>::new();
        for attachment in depth_stencil_attachments {
            depth_stencil_targets.insert(
                attachment,
                (
                    create_target::<B, _, I, E>(
                        attachment.format,
                        &mut allocator,
                        device,
                        &mut images,
                        &mut image_views,
                        self.extent,
                        frames,
                        true,
                    ).map_err(GraphBuildError::AllocationError)?,
                    0,
                ),
            );
        }

        info!("Build pass nodes from pass builders");
        let mut pass_nodes: Vec<PassNode<B, T>> = Vec::new();

        let mut first_draws_to_surface = None;

        for (pass, last_dep) in passes.into_iter().zip(deps) {
            info!("Collect samped inputs");
            let sampled = pass.sampled
                .iter()
                .map(|input| {
                    let (ref indices, ref written) = *match *input {
                        Attachment::Color(ref color) => &color_targets[&color.ptr()],
                        Attachment::DepthStencil(ref depth_stencil) => {
                            &depth_stencil_targets[&depth_stencil.ptr()]
                        }
                    };
                    let indices = indices.clone();
                    debug_assert!(*written > 0);
                    InputAttachmentDesc {
                        format: input.format(),
                        view: indices,
                    }
                })
                .collect::<Vec<_>>();

            info!("Collect input targets");
            let inputs = pass.inputs
                .iter()
                .map(|input| {
                    let (ref indices, ref written) = *match *input {
                        Attachment::Color(ref color) => &color_targets[&color.ptr()],
                        Attachment::DepthStencil(ref depth_stencil) => {
                            &depth_stencil_targets[&depth_stencil.ptr()]
                        }
                    };
                    let indices = indices.clone();
                    debug_assert!(*written > 0);
                    InputAttachmentDesc {
                        format: input.format(),
                        view: indices,
                    }
                })
                .collect::<Vec<_>>();

            info!("Collect color targets");
            let colors = pass.colors
                .iter()
                .enumerate()
                .map(|(index, &color)| {
                    if first_draws_to_surface.is_none() && eq(color, present) {
                        first_draws_to_surface = Some(index);
                    }
                    let (ref indices, ref mut written) =
                        *color_targets.get_mut(&color.ptr()).unwrap();
                    let indices = indices.clone();
                    let clear = if *written == 0 { color.clear } else { None };

                    *written += 1;

                    ColorAttachmentDesc {
                        format: color.format,
                        view: if is_images(backbuffer) {
                            AttachmentImageViews::Owned(indices)
                        } else {
                            AttachmentImageViews::External
                        },
                        clear,
                    }
                })
                .collect::<Vec<_>>();

            info!("Collect depth-stencil targets");
            let depth_stencil = pass.depth_stencil.clone().map(|depth_stencil| {
                let (ref indices, ref mut written) =
                    *depth_stencil_targets.get_mut(&depth_stencil.ptr()).unwrap();
                let indices = indices.clone();
                let clear = if *written == 0 { depth_stencil.clear } else { None };

                *written += 1;

                DepthStencilAttachmentDesc {
                    format: depth_stencil.format,
                    view: if is_images(backbuffer) {
                        AttachmentImageViews::Owned(indices)
                    } else {
                        AttachmentImageViews::External
                    },
                    clear,
                }
            });

            let mut node =
                pass.build(device, &sampled, &inputs, &colors, depth_stencil, self.extent, &image_views)?;

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
            first_draws_to_surface: first_draws_to_surface.unwrap(),
        })
    }
}

fn reorder_passes<'a, B, T: 'a>(
    mut unscheduled: Vec<PassBuilder<'a, B, T>>,
) -> (Vec<PassBuilder<'a, B, T>>, Vec<Option<usize>>)
where
    B: Backend,
{
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

/// Get all color attachments for all passes
fn color_attachments<'a, B, T>(passes: &[PassBuilder<'a, B, T>]) -> Vec<&'a ColorAttachment>
where
    B: Backend,
{
    let mut attachments = Vec::new();
    for pass in passes {
        attachments.extend(pass.colors.iter().cloned());
    }
    attachments.sort_by_key(|a| a as *const _);
    attachments.dedup_by_key(|a| a as *const _);
    attachments
}

/// Get all depth_stencil attachments for all passes
fn depth_stencil_attachments<'a, B, T>(
    passes: &[PassBuilder<'a, B, T>],
) -> Vec<&'a DepthStencilAttachment>
where
    B: Backend,
{
    let mut attachments = Vec::new();
    for pass in passes {
        attachments.extend(pass.depth_stencil.as_ref());
    }
    attachments.sort_by_key(|a| a as *const _);
    attachments.dedup_by_key(|a| a as *const _);
    attachments
}

fn create_target<B, A, I, E>(
    format: Format,
    mut allocator: A,
    device: &B::Device,
    images: &mut Vec<I>,
    views: &mut Vec<B::ImageView>,
    extent: Extent,
    frames: usize,
    depth: bool,
) -> Result<Range<usize>, E>
where
    B: Backend,
    A: FnMut(Kind, Level, Format, Usage, Properties, &B::Device) -> Result<I, E>,
    I: Borrow<B::Image>,
{
    let kind = Kind::D2(extent.width as u16, extent.height as u16, AaMode::Single);
    let start = views.len();
    for _ in 0..frames {
        let image = allocator(
            kind,
            1,
            format,
            if depth {
                Usage::DEPTH_STENCIL_ATTACHMENT
            } else {
                Usage::COLOR_ATTACHMENT
            },
            Properties::DEVICE_LOCAL,
            device,
        )?;
        let view = device
            .create_image_view(image.borrow(), format, Swizzle::NO, COLOR_RANGE.clone())
            .expect("Views are expected to be created");
        views.push(view);
        images.push(image);
    }
    Ok(start..views.len())
}

/// Get dependencies of pass.
fn direct_dependencies<'a, B, T>(
    passes: &'a [PassBuilder<'a, B, T>],
    pass: &'a PassBuilder<'a, B, T>,
) -> Vec<usize>
where
    B: Backend,
{
    let mut deps = Vec::new();
    for input in pass.inputs.iter().chain(&pass.sampled) {
        deps.extend(
            passes
                .iter()
                .enumerate()
                .filter(|p| {
                    p.1
                        .depth_stencil
                        .as_ref()
                        .map_or(false, |&a| input.is(Attachment::DepthStencil(a)))
                        || p.1
                            .colors
                            .iter()
                            .any(|&a| input.is(Attachment::Color(a)))
                })
                .map(|p| p.0),
        );
    }
    deps.sort();
    deps.dedup();
    deps
}

/// Get other passes that shares output attachments
fn siblings<'a, B, T>(
    passes: &'a [PassBuilder<'a, B, T>],
    pass: &'a PassBuilder<'a, B, T>,
) -> Vec<usize>
where
    B: Backend,
{
    let mut siblings = Vec::new();
    for &color in pass.colors.iter() {
        siblings.extend(
            passes
                .iter()
                .enumerate()
                .filter(|p| p.1.colors.iter().any(|&a| eq(a, color)))
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
                        .map_or(false, |&a| eq(a, depth_stencil))
                })
                .map(|p| p.0),
        );
    }
    siblings.sort();
    siblings.dedup();
    siblings
}

/// Get dependencies of pass. And dependencies of dependencies.
fn dependencies<'a, B, T>(
    passes: &'a [PassBuilder<'a, B, T>],
    pass: &'a PassBuilder<'a, B, T>,
) -> Vec<usize>
where
    B: Backend,
{
    let mut deps = direct_dependencies(passes, pass);
    deps = deps.into_iter()
        .flat_map(|dep| dependencies(passes, &passes[dep]))
        .collect();
    deps.sort();
    deps.dedup();
    deps
}

/// Check if backbuffer is a collection of images
/// or external framebuffer.
fn is_images<B>(backbuffer: &Backbuffer<B>) -> bool
where
    B: Backend,
{
    match *backbuffer {
        Backbuffer::Images(_) => true,
        Backbuffer::Framebuffer(_) => false,
    }
}
