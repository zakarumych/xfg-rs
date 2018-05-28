use std::{borrow::Borrow, ops::Range};

use hal::{
    buffer, format::Format, image, pool::{CommandPool, CommandPoolCreateFlags}, pso::PipelineStage,
    queue::{Capability, CommandQueue}, window::Backbuffer, Backend, Device,
};

use self::build::NodeBuilder;
use util::*;

pub mod build;
pub mod low;
pub mod present;
pub mod render;

/// Overall description for node.
pub trait NodeDesc: Send + Sync + Sized + 'static {
    /// Iterator of image state, usage and stages where image is used.
    type Buffers: IntoIterator<Item = (buffer::Usage, buffer::State, PipelineStage)>;

    /// Iterator of image state, usage and stages where image is used.
    type Images: IntoIterator<Item = (image::Usage, image::State, PipelineStage)>;

    /// Type of command buffer capabilities required for the node.
    type Capability: Capability;

    /// Name of this node.
    fn name() -> &'static str;

    /// Buffers used by node.
    fn buffers() -> Self::Buffers;

    /// Images used by node.
    fn images() -> Self::Images;

    /// Create `NodeBuilder` for this node.
    fn builder() -> NodeBuilder<Self> {
        NodeBuilder::new()
    }
}

/// Graph node - building block of the graph
pub trait Node<B, D, T>: NodeDesc
where
    B: Backend,
    D: Device<B>,
{
    /// Build node.
    ///
    /// # Parameters
    ///
    /// `buffers`   - Information about buffers. One for each returned by `buffers` function.
    ///
    /// `images`    - Information about images. One for each returned by `images` function.
    ///
    /// `frames`     - number of frames in swapchain. All non-read-only resources must be allocated per frame.
    ///               `frame` argument of `run` method will be always in `0 .. frames`.
    ///
    /// `device`    - `Device<B>` implementation. `B::Device` or wrapper.
    ///
    /// `aux`       - auxiliary data container. May be anything the implementation desires.
    ///
    /// `pools`     - function to allocate command pools compatible with queue assigned to the node.
    ///
    /// This methods builds node instance and returns it.
    fn build<F, U, I>(
        buffers: Vec<BufferInfo<U>>,
        images: Vec<ImageInfo<I>>,
        pools: F,
        device: &mut D,
        aux: &mut T,
    ) -> Self
    where
        F: FnMut(&mut D, CommandPoolCreateFlags) -> CommandPool<B, Self::Capability>,
        U: Borrow<B::Buffer>,
        I: Borrow<B::Image>;

    /// Record commands for the node and return them as `Submit` object.
    /// `frame`     - index of the frame for which commands are recorded.
    ///               Node can safely reuse resources used with same frame last time.
    ///
    /// `device`    - `Device<B>` implementation. `B::Device` or wrapper.
    ///
    /// `aux`       - auxiliary data container. May be anything the implementation desires.
    ///
    /// This method returns iterator of submittables with primary level and capabilities declared by node.
    fn run<'a, W, S>(
        &'a mut self,
        wait: W,
        queue: &mut CommandQueue<B, Self::Capability>,
        signal: S,
        fence: Option<&B::Fence>,
        device: &mut D,
        aux: &'a T,
    ) where
        W: IntoIterator<Item = (&'a B::Semaphore, PipelineStage)>,
        S: IntoIterator<Item = &'a B::Semaphore>;

    /// Dispose of the node.
    fn dispose(self, device: &mut D, aux: &mut T);
}
