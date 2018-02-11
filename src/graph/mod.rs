//!
//! Defines a directed acyclic rendering `Graph`.
//!
//! A rendering `Graph` is built up from a list of `Pass`es, a `Backbuffer`, an `Extent`, and a
//! draw surface. See `GraphBuilder` for more information about creating graphs.
//!
//! TODO: compute.
//!

pub use self::build::{GraphBuildError, GraphBuilder};

use std::ops::Range;

use gfx_hal::{Backend, Device};
use gfx_hal::command::{OneShot, Viewport};
use gfx_hal::pool::CommandPool;
use gfx_hal::pso::PipelineStage;
use gfx_hal::queue::CommandQueue;
use gfx_hal::queue::capability::{Graphics, Supports, Transfer};

use smallvec::SmallVec;

use frame::SuperFrame;
use pass::{Pass, PassNode};

mod build;

/// Directed acyclic rendering graph.
///
/// It contains all data needed for preparing and drawing frames using the graph. Auxiliary data
/// used by the passes should be supplied from the outside for each frame.
///
/// ### Type parameters:
///
/// - `B`: render `Backend`
/// - `I`: render target image type
/// - `T`: auxiliary data used by the graph, user supplied, not touched by the graph itself, only
///        passed on to the `Pass`es
#[derive(Debug)]
pub struct Graph<B: Backend, I, P> {
    passes: Vec<PassNode<B, P>>,
    signals: Vec<Option<B::Semaphore>>,
    images: Vec<I>,
    views: Vec<B::ImageView>,
    frames: usize,
    draws_to_surface: Range<usize>,
}

impl<B, I, P> Graph<B, I, P>
where
    B: Backend,
{
    /// Start building the render graph
    pub fn build() -> GraphBuilder<P> {
        GraphBuilder::new()
    }

    /// Get number of frames that can be rendered in parallel with this graph
    pub fn get_frames_number(&self) -> usize {
        self.frames
    }

    /// Walk over graph recording drawing commands and submitting them to `queue`.
    /// This function handles synchronization between dependent rendering nodes.
    ///
    /// `queue` must come from same `QueueGroup` with which `pool` is associated.
    /// All those should be created by `device`.
    ///
    /// ### Parameters
    ///
    /// - `queue`: queue that commands will be submitted to
    /// - `pool`: command buffer pool to use
    /// - `frame`: frame index that should be drawn.
    ///             (or `Framebuffer` reference that corresponds to index `0`)
    /// - `acquire`: surface acquisition semaphore.
    /// - `release`: presentation will wait on this.
    /// - `viewport`: portion of framebuffers to draw to.
    /// - `finish`: last submission should set this fence.
    ///             Application must wait for this fence before calling `draw_inline` again with same `frame`.
    /// - `device`: you need this guy everywhere =^_^=
    /// - `aux`: auxiliary data for passes.
    ///
    /// ### Type parameters:
    ///
    /// - `C`: hal `Capability`
    pub fn draw_inline<C, T>(
        &mut self,
        queue: &mut CommandQueue<B, C>,
        pool: &mut CommandPool<B, C>,
        frame: SuperFrame<B>,
        acquire: &B::Semaphore,
        release: &B::Semaphore,
        viewport: Viewport,
        finish: &B::Fence,
        device: &B::Device,
        aux: &mut T,
    ) where
        C: Supports<Graphics> + Supports<Transfer>,
        P: Pass<B, T>,
    {
        use gfx_hal::queue::submission::Submission;

        let ref signals = self.signals;
        let count = self.passes.len();
        let ref draws_to_surface = self.draws_to_surface;

        // Record commands for all passes
        self.passes.iter_mut().enumerate().for_each(|(id, pass)| {
            // Pick buffer
            let mut cbuf = pool.acquire_command_buffer::<OneShot>(false);

            // Setup
            cbuf.set_viewports(&[viewport.clone()]);
            cbuf.set_scissors(&[viewport.rect]);

            // Record commands for pass
            pass.prepare(&mut cbuf, device, frame, aux);
            pass.draw_inline(&mut cbuf, device, viewport.rect, frame, aux);

            {
                // If it renders to acquired image
                let wait_surface = if id == draws_to_surface.start {
                    // And it should wait for acquisition
                    Some((acquire, PipelineStage::TOP_OF_PIPE))
                } else {
                    None
                };

                let to_wait = pass.depends
                    .as_ref()
                    .map(|&(id, stage)| (signals[id].as_ref().unwrap(), stage))
                    .into_iter()
                    .chain(wait_surface)
                    .collect::<SmallVec<[_; 3]>>();

                let mut to_signal = SmallVec::<[_; 1]>::new();
                if id == draws_to_surface.end {
                    to_signal.push(release);
                }
                if let Some(signal) = signals[id].as_ref() {
                    to_signal.push(signal);
                }

                // Signal the finish fence in last submission
                let fence = if id == count - 1 { Some(finish) } else { None };

                // Submit buffer
                queue.submit(
                    Submission::new()
                        .promote::<C>()
                        .submit(Some(cbuf.finish()))
                        .wait_on(&to_wait)
                        .signal(&to_signal),
                    fence,
                );
            }
        });
    }

    /// Dispose of the graph, will call `dispose` on all passes, and clean up any `Semaphore`s,
    /// `ImageView`s, and images created by the `Graph` or `GraphBuilder`.
    ///
    /// ### Parameters:
    ///
    /// - `deallocator`: used for image deallocation, should match up with the `allocator` used in
    ///                  `GraphBuilder::build`.
    /// - `device`: graphics device
    /// - `aux`: auxiliary data used by the `Graph` and the `Pass`es in the graph.
    ///
    /// ### Type parameters:
    ///
    /// - `F`: deallocator function
    pub fn dispose<F, T>(self, mut deallocator: F, device: &B::Device, aux: &mut T)
    where
        F: FnMut(I, &B::Device),
        P: Pass<B, T>,
    {
        for pass in self.passes {
            pass.dispose(device, aux);
        }
        for signal in self.signals.into_iter().filter_map(|x| x) {
            device.destroy_semaphore(signal);
        }
        for view in self.views {
            device.destroy_image_view(view);
        }
        for image in self.images {
            deallocator(image, device);
        }
    }
}
