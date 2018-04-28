
use std::marker::PhantomData;

use hal::{Backend, Device,
          buffer,
          image,
          pool::{CommandPool, CommandPoolCreateFlags},
          pso::PipelineStage,
          command::{CommandBuffer, Submit, Secondary, OneShot, Submittable},
          queue::{Capability, QueueFamily},
          };

use chain::{pass::{Pass, PassId, StateUsage},
            resource::{Buffer, Image, Id, State, BufferLayout},
            sync::Sync,
            };

use pick_queue_family;

/// Overall description for node.
pub trait NodeDesc: Sized + 'static {
    /// Buffers used by node.
    const BUFFERS: &'static [(buffer::Usage, buffer::Access, PipelineStage)];

    /// Images used by node.
    const IMAGES: &'static [(image::Usage, image::Access, image::Layout, PipelineStage)];

    /// Type of command buffer capabilities required for the node.
    type Capability: Capability;
}

/// Graph node - building block of the graph
pub trait Node<B, D, T>: NodeDesc
where
    B: Backend,
{
    /// Build node.
    /// 
    /// # Parameters
    /// 
    /// `images`    - actual images layout for images from `IMAGES` associated const.
    /// `frame`     - number of frames in swapchain. All non-read-only resources must be allocated per frame.
    /// `device`    - `hal::Device<B>` implementation.
    /// `aux`       - auxiliary data container. May be anything the implementation desires. For example `&specs::World`.
    /// `pools`     - function that can be used to allocate command pools. Those pools are compatible with queue in `run` method.
    /// 
    fn build<F>(images: &[image::Layout], frames: usize, pools: F, device: &mut D, aux: &mut T) -> Self
    where
        F: FnMut(&mut D, CommandPoolCreateFlags) -> CommandPool<B, Self::Capability>;

    /// Record commands for the node and submit them to the queue.
    fn run(&mut self, frame: usize, device: &mut D, aux: &T) -> Submit<B, Self::Capability, OneShot, Secondary>;
}

pub(crate) trait AnyNode<B, D, T>
where
    B: Backend,
{
    fn run(&mut self, frame: usize, device: &mut D, aux: &T) -> B::CommandBuffer;
}

impl<B, D, T, N> AnyNode<B, D, T> for (N,)
where
    B: Backend,
    N: Node<B, D, T>,
{
    fn run(&mut self, frame: usize, device: &mut D, aux: &T) -> B::CommandBuffer {
        let submit = N::run(&mut self.0, frame, device, aux);
        unsafe {
            submit.into_buffer().into_owned()
        }
    }
}

impl<B, D, T, N> AnyNode<B, D, T> for Box<N>
where
    B: Backend,
    N: AnyNode<B, D, T> + ?Sized,
{
    fn run(&mut self, frame: usize, device: &mut D, aux: &T) -> B::CommandBuffer {
        AnyNode::run(&mut**self, frame, device, aux)
    }
}

trait AnyDesc<B>
where
    B: Backend,
{
    fn chain(&self, id: PassId, families: &[B::QueueFamily], buffers: &[Id<Buffer>], images: &[Id<Image>], dependencies: Vec<PassId>) -> Pass;
}

trait AnyBuilder<B, D, T>: AnyDesc<B>
where
    B: Backend,
{
    fn build(&self, images: &[image::Layout], frames: usize, family: &B::QueueFamily, device: &mut D, aux: &mut T) -> Box<AnyNode<B, D, T>>;
}

impl<B, N> AnyDesc<B> for PhantomData<N>
where
    B: Backend,
    N: NodeDesc,
{
    fn chain(&self, id: PassId, families: &[B::QueueFamily], buffers: &[Id<Buffer>], images: &[Id<Image>], dependencies: Vec<PassId>) -> Pass {
        assert_eq!(buffers.len(), N::BUFFERS.len());
        assert_eq!(images.len(), N::IMAGES.len());

        Pass {
            id,
            family: pick_queue_family::<B, N::Capability, _>(families),
            queue: None,
            dependencies,
            buffers: buffers.iter().cloned().zip(N::BUFFERS.iter().cloned().map(buffer_state_usage)).collect(),
            images: images.iter().cloned().zip(N::IMAGES.iter().cloned().map(image_state_usage)).collect(),
        }
    }
}

impl<B, D, T, N> AnyBuilder<B, D, T> for PhantomData<N>
where
    B: Backend,
    D: Device<B>,
    N: Node<B, D, T>,
{
    fn build(&self, images: &[image::Layout], frames: usize, family: &B::QueueFamily, device: &mut D, aux: &mut T) -> Box<AnyNode<B, D, T>> {
        let node = <N as Node<B, D, T>>::build(images, frames, |device, flags| create_typed_pool(family, flags, device), device, aux);
        Box::new((node,))
    }
}

pub(crate) struct Info<B: Backend, D, T> {
    builder: Box<AnyBuilder<B, D, T>>,
    buffers: Vec<Id<Buffer>>,
    images: Vec<Id<Image>>,
    dependencies: Vec<PassId>,
}

impl<B, D, T> Info<B, D, T>
where
    B: Backend,
    D: Device<B>,
{
    pub(crate) fn new<N>() -> Self
    where
        N: Node<B, D, T>,
    {
        Info {
            builder: Box::new(PhantomData::<N>),
            buffers: Vec::new(),
            images: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    pub(crate) fn chain(&self, index: usize, families: &[B::QueueFamily]) -> Pass {
        self.builder.chain(PassId(index), families, &self.buffers, &self.images, self.dependencies.clone())
    }

    pub(crate) fn build(&self, images: &[image::Layout], frames: usize, family: &B::QueueFamily, device: &mut D, aux: &mut T) -> Box<AnyNode<B, D, T>> {
        self.builder.build(images, frames, family, device, aux)
    }
}

fn create_typed_pool<B, D, C>(family: &B::QueueFamily, flags: CommandPoolCreateFlags, device: &mut D) -> CommandPool<B, C>
where
    B: Backend,
    D: Device<B>,
    C: Capability,
{
    assert!(C::supported_by(family.queue_type()));
    unsafe {
        CommandPool::new(device.create_command_pool(family.id(), flags))
    }
}

fn buffer_state_usage((usage, access, stages): (buffer::Usage, buffer::Access, PipelineStage)) -> StateUsage<Buffer> {
    StateUsage {
        state: State {
            access,
            layout: BufferLayout,
            stages,
        },
        usage,
    }
}

fn image_state_usage((usage, access, layout, stages): (image::Usage, image::Access, image::Layout, PipelineStage)) -> StateUsage<Image> {
    StateUsage{
        state: State {
            access,
            layout,
            stages,
        },
        usage,
    }
}
