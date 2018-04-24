
use hal::{pool, command, queue};

/// Graph node - building block of the graph
pub trait Node<B: Backend, D, T>: Sized {
    /// Number of graph's images used by this node.
    const IMAGES: usize;

    /// Number of graph's buffers used by this node.
    const BUFFERS: usize;

    /// Type of command buffer capabilities required for the node.
    type Capability: queue::Capability;

    /// Build node.
    /// 
    /// # Parameters
    /// 
    /// `frame`     - number of frames in swapchain. All non-read-only resources must be allocated per frame.
    /// `device`    - `hal::Device<B>` implementation.
    /// `aux`       - auxiliary data container. May be anything the implementation desires. For example `&specs::World`.
    /// `pools`     - function that can be used to allocate command pools. Those pools are compatible with queue in `run` method.
    /// 
    fn build<F>(frames: usize, device: &mut D, aux: &mut T, pools: F) -> Self
    where
        F: FnMut(&mut D, pool::CommandPoolCreateFlags) -> CommandPool<B, Self::Capability>;

    /// Record commands for the node.
    fn run<L>(&mut self, frame: usize, buffers: &[&B::Buffer; Self::BUFFERS], images: &[&B::Image; Self::IMAGES], device: &mut D, aux: &T, queue: &mut CommandQueue<B, Self::Capability>)
    where
        L: Level;
}

trait AnyNode<B, D, T> {
    fn images() -> usize;
    fn buffers() -> usize;

    fn build(frames: usize, device: &mut D, aux: &mut T, family: &B::QueueFamily) -> Box<Self>
    where
        Self: Sized;

    fn run(&mut self, buffers: &B::Buffer, images: &B::Image, device: &mut D, aux: &mut T) -> B::CommandBuffer;
}


impl<B, D, T, N> AnyNode<B, D, T> for N
where
    B: Backend,
    D: Device<B>,
    N: Node<B, D, T>,
{
    fn images() -> usize {
        N::IMAGES
    }
    fn buffers() -> usize {
        N::BUFFERS
    }

    fn build(frames: usize, device: &mut D, aux: &mut T, family: &B::QueueFamily) -> Box<N>
    where
        N: Sized,
    {
        assert!(N::Capablity::supported_by(family.queue_type()));
    }
}

