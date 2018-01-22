//!
//! Defines `Graph` - complex rendering graph.
//! And `Pass` - building block for `Graph`.
//! TODO: compute.
//! 

mod build;
use std::ops::Range;

use gfx_hal::{Backend, Device};
use gfx_hal::command::Viewport;
use gfx_hal::device::{FramebufferError, ShaderError};
use gfx_hal::pool::CommandPool;
use gfx_hal::pso::PipelineStage;
use gfx_hal::queue::CommandQueue;
use gfx_hal::queue::capability::{Graphics, Supports, Transfer};

use smallvec::SmallVec;

use epoch::Epoch;
use frame::SuperFrame;
use pass::PassNode;

pub use self::build::GraphBuilder;

#[derive(Fail, Debug, Clone)]
pub enum Error {
    #[fail(display = "Failed to create framebuffer")] FramebufferError,

    #[fail(display = "Shader compilation failed: {}", msg)] CompilationFailed { msg: String },

    #[fail(display = "Missing shader entry point: {}", msg)] MissingEntryPoint { msg: String },

    #[fail(display = "Shader interface mismatch: {}", msg)] InterfaceMismatch { msg: String },
}

impl From<FramebufferError> for Error {
    fn from(_: FramebufferError) -> Error {
        Error::FramebufferError
    }
}

impl From<ShaderError> for Error {
    fn from(error: ShaderError) -> Error {
        match error {
            ShaderError::CompilationFailed(msg) => Error::CompilationFailed { msg }.into(),
            ShaderError::MissingEntryPoint(msg) => Error::MissingEntryPoint { msg }.into(),
            ShaderError::InterfaceMismatch(msg) => Error::InterfaceMismatch { msg }.into(),
        }
    }
}


/// Directed acyclic rendering graph.
/// It holds all rendering nodes and auxiliary data.
#[derive(Debug)]
pub struct Graph<B: Backend, I, T> {
    passes: Vec<PassNode<B, T>>,
    signals: Vec<Option<B::Semaphore>>,
    images: Vec<I>,
    views: Vec<B::ImageView>,
    frames: usize,
    first_draws_to_surface: usize,
}

impl<B, I, T> Graph<B, I, T>
where
    B: Backend,
{
    pub fn build<'a>() -> GraphBuilder<'a, B, T> {
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
    /// `frame` - frame index that should be drawn.
    /// `upload` - semaphore that will be signaled when all data will be uploaded.
    /// (or `Framebuffer` reference that corresponds to index `0`)
    /// `acquire` - semaphore that should be waited on by submissions which
    /// contains commands from passes that draw to the surface
    /// `device` - you need this guy everywhere =^_^=
    /// `viewport` - portion of framebuffers to draw
    /// `res` - primary source of stuff to draw
    /// `finish` - last submission should set this fence
    /// `span` - all commands will be finished before this epoch ends.
    pub fn draw_inline<C>(
        &mut self,
        span: Range<Epoch>,
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
    {
        use gfx_hal::queue::submission::Submission;

        let ref signals = self.signals;
        let count = self.passes.len();
        let first_draws_to_surface = self.first_draws_to_surface;

        // Record commands for all passes
        self.passes.iter_mut().enumerate().for_each(|(id, pass)| {
            // Pick buffer
            let mut cbuf = pool.acquire_command_buffer();

            // Setup
            cbuf.set_viewports(&[viewport.clone()]);
            cbuf.set_scissors(&[viewport.rect]);

            // Record commands for pass
            pass.draw_inline(
                span.clone(),
                &mut cbuf,
                viewport.rect,
                frame.clone(),
                device,
                aux,
            );

            {
                // If it renders to acquired image
                let wait_surface = if id == first_draws_to_surface {
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
                if id == count - 1 {
                    // The last one has to draw to surface.
                    // Also it depends on all others that draws to surface.
                    to_signal.push(release);
                } else if let Some(signal) = signals[id].as_ref() {
                    to_signal.push(signal);
                };

                // Signal the finish fence in last submission
                let fence = if id == count - 1 { Some(finish) } else { None };

                // Submit buffer
                queue.submit(
                    Submission::new()
                        .promote::<C>()
                        .submit(&[cbuf.finish()])
                        .wait_on(&to_wait)
                        .signal(&to_signal),
                    fence,
                );
            }
        });
    }

    pub fn dispose<F>(self, mut deallocator: F, device: &B::Device, aux: &mut T)
    where
        F: FnMut(I, &B::Device),
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
