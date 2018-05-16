use std::{borrow::{Borrow, Cow}, cmp::Ordering, marker::PhantomData, ops::Range};

use hal::{buffer, image, Backend, Device, command::Submittable,
          pool::{CommandPool, CommandPoolCreateFlags}, pso::PipelineStage,
          queue::{Capability, QueueFamily, QueueFamilyId, QueueType}};

use chain::{pass::{Pass, PassId, StateUsage}, resource::{Buffer, BufferLayout, Image, State},
            sync::{Guard, SyncData}};

use smallvec::SmallVec;

use id::{BufferId, ImageId};
use node::{Barriers, Node, NodeDesc};

pub trait AnyNode<B, D, T>: Send + Sync
where
    B: Backend,
{
    fn run<'a>(
        &'a mut self,
        frame: usize,
        device: &mut D,
        aux: &'a T,
        extend: &mut SmallVec<[Cow<'a, B::CommandBuffer>; 4]>,
    );
}

impl<B, D, T, N> AnyNode<B, D, T> for (N,)
where
    B: Backend,
    N: Node<B, D, T>,
{
    fn run<'a>(
        &'a mut self,
        frame: usize,
        device: &mut D,
        aux: &'a T,
        extend: &mut SmallVec<[Cow<'a, B::CommandBuffer>; 4]>,
    ) {
        extend.extend(N::run(&mut self.0, frame, device, aux).map(|s| unsafe { s.into_buffer() }))
    }
}

trait AnyDesc<B>: Send + Sync
where
    B: Backend,
{
    fn chain(
        &self,
        id: PassId,
        families: &[&B::QueueFamily],
        buffers: &[BufferId],
        images: &[ImageId],
        dependencies: Vec<PassId>,
    ) -> Pass;
}

trait AnyNodeBuilder<B, D, T>: AnyDesc<B>
where
    B: Backend,
{
    fn build(
        &self,
        acquire: Barriers,
        release: Barriers,
        buffers: Vec<BufferId>,
        images: Vec<ImageId>,
        frames: usize,
        family: &B::QueueFamily,
        device: &mut D,
        aux: &mut T,
    ) -> Box<AnyNode<B, D, T>>;
}

impl<B, N> AnyDesc<B> for PhantomData<N>
where
    B: Backend,
    N: NodeDesc,
{
    fn chain(
        &self,
        id: PassId,
        families: &[&B::QueueFamily],
        buffers: &[BufferId],
        images: &[ImageId],
        dependencies: Vec<PassId>,
    ) -> Pass {
        assert_eq!(buffers.len(), N::BUFFERS.len());
        assert_eq!(images.len(), N::IMAGES.len());

        Pass {
            id,
            family: pick_queue_family::<B, N::Capability, _>(families.iter().cloned()),
            queue: None,
            dependencies,
            buffers: buffers
                .iter()
                .map(|id| id.0)
                .zip(N::BUFFERS.iter().cloned().map(buffer_state_usage))
                .collect(),
            images: images
                .iter()
                .map(|id| id.0)
                .zip(N::IMAGES.iter().cloned().map(image_state_usage))
                .collect(),
        }
    }
}

impl<B, D, T, N> AnyNodeBuilder<B, D, T> for PhantomData<N>
where
    B: Backend,
    D: Device<B>,
    N: Node<B, D, T>,
{
    fn build(
        &self,
        acquire: Barriers,
        release: Barriers,
        buffers: Vec<BufferId>,
        images: Vec<ImageId>,
        frames: usize,
        family: &B::QueueFamily,
        device: &mut D,
        aux: &mut T,
    ) -> Box<AnyNode<B, D, T>> {
        let node = N::build(
            acquire,
            release,
            buffers,
            images,
            frames,
            |device, flags| create_typed_pool(family, flags, device),
            device,
            aux,
        );
        Box::new((node,))
    }
}

pub struct NodeBuilder<B: Backend, D, T> {
    builder: Box<AnyNodeBuilder<B, D, T>>,
    buffers: Vec<BufferId>,
    images: Vec<ImageId>,
    dependencies: Vec<PassId>,
}

impl<B, D, T> NodeBuilder<B, D, T>
where
    B: Backend,
    D: Device<B>,
{
    /// Create `NodeBuilder` that builds node of type `N`.
    pub fn new<N>() -> Self
    where
        N: Node<B, D, T>,
    {
        NodeBuilder {
            builder: Box::new(PhantomData::<N>),
            buffers: Vec::new(),
            images: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    /// Add buffer id.
    /// This id will be associated with next buffer from `Node::BUFFERS` starting from `0`.
    pub fn add_buffer(&mut self, id: BufferId) -> &mut Self {
        self.buffers.push(id);
        self
    }

    /// Add buffer id.
    /// This id will be associated with next buffer from `Node::BUFFERS` starting from `0`.
    pub fn with_buffer(mut self, id: BufferId) -> Self {
        self.add_buffer(id);
        self
    }

    /// Add image id.
    /// This id will be associated with next image from `Node::IMAGES` starting from `0`.
    pub fn add_image(&mut self, id: ImageId) -> &mut Self {
        self.images.push(id);
        self
    }

    /// Add image id.
    /// This id will be associated with next image from `Node::IMAGES` starting from `0`.
    pub fn with_image(mut self, id: ImageId) -> Self {
        self.add_image(id);
        self
    }

    pub(crate) fn chain(&self, index: usize, families: &[&B::QueueFamily]) -> Pass {
        self.builder.chain(
            PassId(index),
            families,
            &self.buffers,
            &self.images,
            self.dependencies.clone(),
        )
    }

    pub(crate) fn build<S, W>(
        &self,
        sync: &SyncData<S, W>,
        frames: usize,
        family: &B::QueueFamily,
        device: &mut D,
        aux: &mut T,
    ) -> Box<AnyNode<B, D, T>> {
        let buffers = self.buffers.iter().cloned();
        let images = self.images.iter().cloned();

        let acquire = barriers(buffers.clone(), images.clone(), &sync.acquire);
        let release = barriers(buffers.clone(), images.clone(), &sync.release);

        self.builder.build(
            acquire,
            release,
            self.buffers.clone(),
            self.images.clone(),
            frames,
            family,
            device,
            aux,
        )
    }
}

fn create_typed_pool<B, D, C>(
    family: &B::QueueFamily,
    flags: CommandPoolCreateFlags,
    device: &mut D,
) -> CommandPool<B, C>
where
    B: Backend,
    D: Device<B>,
    C: Capability,
{
    assert!(C::supported_by(family.queue_type()));
    unsafe { CommandPool::new(device.create_command_pool(family.id(), flags)) }
}

fn buffer_state_usage(
    (usage, access, stages): (buffer::Usage, buffer::Access, PipelineStage),
) -> StateUsage<Buffer> {
    StateUsage {
        state: State {
            access,
            layout: BufferLayout,
            stages,
        },
        usage,
    }
}

fn image_state_usage(
    (usage, (access, layout), stages): (
        image::Usage,
        (image::Access, image::Layout),
        PipelineStage,
    ),
) -> StateUsage<Image> {
    StateUsage {
        state: State {
            access,
            layout,
            stages,
        },
        usage,
    }
}

fn pick_queue_family<B, C, F>(families: F) -> QueueFamilyId
where
    B: Backend,
    C: Capability,
    F: IntoIterator,
    F::Item: Borrow<B::QueueFamily>,
{
    families
        .into_iter()
        .filter(|qf| C::supported_by(qf.borrow().queue_type()))
        .min_by(
            |left, right| match (left.borrow().queue_type(), right.borrow().queue_type()) {
                (_, QueueType::General) => Ordering::Less,
                (QueueType::General, _) => Ordering::Greater,
                (QueueType::Transfer, _) => Ordering::Less,
                _ => Ordering::Equal,
            },
        )
        .unwrap()
        .borrow()
        .id()
}

fn barriers<B, I, S, W>(buffers: B, images: I, guard: &Guard<S, W>) -> Barriers
where
    B: Iterator<Item = BufferId>,
    I: Iterator<Item = ImageId>,
{
    Barriers {
        buffers: buffers
            .map(|id| {
                let Range { ref start, ref end } = guard.buffers[&id.0].states;
                (start.access, start.stages)..(end.access, end.stages)
            })
            .collect(),
        images: images
            .map(|id| {
                let Range { ref start, ref end } = guard.images[&id.0].states;
                ((start.access, start.layout), start.stages)..((end.access, end.layout), end.stages)
            })
            .collect(),
    }
}
