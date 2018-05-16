use std::ops::Range;

use hal::{buffer, image, Backend, command::{Primary, Submittable},
          pool::{CommandPool, CommandPoolCreateFlags}, pso::PipelineStage, queue::Capability};

use id::{BufferId, ImageId};

pub mod build;

/// Overall description for node.
pub trait NodeDesc: Send + Sync + Sized + 'static {
    /// Buffers used by node.
    const BUFFERS: &'static [(buffer::Usage, buffer::State, PipelineStage)];

    /// Images used by node.
    const IMAGES: &'static [(image::Usage, image::State, PipelineStage)];

    /// Type of command buffer capabilities required for the node.
    type Capability: Capability;
}

/// With this trait `Node` implementation defines type of iterator of submittables `Node::run` returns.
pub trait Submittables<'a, B: Backend, D, T>: NodeDesc {
    type Submittable: Submittable<'a, B, Self::Capability, Primary>;
    type Iterator: Iterator<Item = Self::Submittable>;
}

/// Graph node - building block of the graph
pub trait Node<B, D, T>: NodeDesc + for<'a> Submittables<'a, B, D, T>
where
    B: Backend,
{
    /// Build node.
    ///
    /// # Parameters
    ///
    /// `acquire`   - barriers that must be recorded before any other commands.
    ///
    /// `release`   - barriers that must be recorded after all other commands.
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
    fn build<F>(
        acquire: Barriers,
        release: Barriers,
        buffers: Vec<BufferId>,
        images: Vec<ImageId>,
        frames: usize,
        pools: F,
        device: &mut D,
        aux: &mut T,
    ) -> Self
    where
        F: FnMut(&mut D, CommandPoolCreateFlags) -> CommandPool<B, Self::Capability>;

    /// Record commands for the node and return them as `Submit` object.
    /// `frame`     - index of the frame for which commands are recorded.
    ///               Node can safely reuse resources used with same frame last time.
    ///
    /// `device`    - `Device<B>` implementation. `B::Device` or wrapper.
    ///
    /// `aux`       - auxiliary data container. May be anything the implementation desires.
    ///
    /// This method returns iterator of submittables with primary level and capabilities declared by node.
    fn run<'a>(
        &'a mut self,
        frame: usize,
        device: &mut D,
        aux: &'a T,
    ) -> <Self as Submittables<'a, B, D, T>>::Iterator;
}

/// Set of barriers for the node to execute.
pub struct Barriers {
    /// Buffer barriers.
    pub buffers: Vec<Range<(buffer::State, PipelineStage)>>,

    /// Image barriers.
    pub images: Vec<Range<(image::State, PipelineStage)>>,
}
