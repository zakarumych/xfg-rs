use std::{borrow::Cow, ops::Range};

use hal::{buffer, image, Backend,
          command::{Level, MultiShot, OneShot, Primary, Submit, Submittable},
          pool::{CommandPool, CommandPoolCreateFlags}, pso::PipelineStage, queue::Capability};

use id::{BufferId, ImageId};

pub mod wrap;

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
}

/// With this trait `Node` implementation defines type of iterator of submittables `Node::run` returns.
pub trait Submittables<'a, B: Backend, D, T>: NodeDesc {
    type Submittable: Submittable<'a, B, Self::Capability, Primary>;
    type IntoIter: IntoIterator<Item = Self::Submittable>;
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
        buffers: Vec<BufferInfo>,
        images: Vec<ImageInfo>,
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
    ) -> <Self as Submittables<'a, B, D, T>>::IntoIter;
}

/// Set of barriers for the node to execute.
pub struct Barriers<S> {
    /// Buffer barriers.
    pub acquire: Range<(S, PipelineStage)>,

    /// Image barriers.
    pub release: Range<(S, PipelineStage)>,
}

/// Image info.
pub struct ImageInfo {
    /// Id of the image.
    pub id: ImageId,

    /// Barriers required for the image.
    pub barriers: Barriers<image::State>,

    /// Layout in which the image is for the node.
    pub layout: image::Layout,
}

/// Buffer info.
pub struct BufferInfo {
    /// Id of the buffer.
    pub id: BufferId,

    /// Barriers required for the buffer.
    pub barriers: Barriers<buffer::State>,
}

/// Convenient enum to hold either `OneShot` command buffer or reference to `MultiShot`.
/// Implements `Submittable`.
pub enum EitherSubmit<'a, B: Backend, C: 'a, L: 'a> {
    /// One shot submit.
    OneShot(Submit<B, C, OneShot, L>),

    /// Multi shot submit reference.
    MultiShot(&'a Submit<B, C, MultiShot, L>),
}

impl<'a, B, C, L> From<Submit<B, C, OneShot, L>> for EitherSubmit<'a, B, C, L>
where
    B: Backend,
{
    fn from(submit: Submit<B, C, OneShot, L>) -> Self {
        EitherSubmit::OneShot(submit)
    }
}

impl<'a, B, C, L> From<&'a Submit<B, C, MultiShot, L>> for EitherSubmit<'a, B, C, L>
where
    B: Backend,
{
    fn from(submit: &'a Submit<B, C, MultiShot, L>) -> Self {
        EitherSubmit::MultiShot(submit)
    }
}

unsafe impl<'a, B, C, L> Submittable<'a, B, C, L> for EitherSubmit<'a, B, C, L>
where
    B: Backend,
    L: Level,
{
    unsafe fn into_buffer(self) -> Cow<'a, B::CommandBuffer> {
        match self {
            EitherSubmit::OneShot(submit) => submit.into_buffer(),
            EitherSubmit::MultiShot(submit) => submit.into_buffer(),
        }
    }
}
