use std::{borrow::{Borrow, Cow}, cmp::Ordering, marker::PhantomData, ops::Range};

use hal::{buffer, image, Backend, Device,
          command::Submittable,
          pool::{CommandPool, CommandPoolCreateFlags}, pso::PipelineStage,
          queue::{Capability, QueueFamily, QueueFamilyId, QueueType}};

use chain::{chain::ImageChains, pass::{Pass, PassId, StateUsage},
            resource::{Buffer, BufferLayout, Image, State}, schedule::Submission, sync::SyncData};

use smallvec::SmallVec;

use id::{BufferId, ImageId};
use node::{Barriers, BufferInfo, ImageInfo, Node, NodeDesc};
use graph::{BufferResource, ImageResource};

pub trait AnyNode<B, D, T>: Send + Sync
where
    B: Backend,
{
    fn name(&self) -> &'static str;

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
    fn name(&self) -> &'static str {
        N::name()
    }

    fn run<'a>(
        &'a mut self,
        frame: usize,
        device: &mut D,
        aux: &'a T,
        extend: &mut SmallVec<[Cow<'a, B::CommandBuffer>; 4]>,
    ) {
        extend.extend(
            N::run(&mut self.0, frame, device, aux)
                .into_iter()
                .map(|s| unsafe { s.into_buffer() }),
        )
    }
}

trait AnyDesc<B>: Send + Sync
where
    B: Backend,
{
    fn name(&self) -> &str;

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
        buffers: Vec<BufferInfo<B>>,
        images: Vec<ImageInfo<B>>,
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
    fn name(&self) -> &str {
        N::name()
    }

    fn chain(
        &self,
        id: PassId,
        families: &[&B::QueueFamily],
        buffers: &[BufferId],
        images: &[ImageId],
        dependencies: Vec<PassId>,
    ) -> Pass {
        let pass = Pass {
            id,
            family: pick_queue_family::<B, N::Capability, _>(families.iter().cloned()),
            queue: None,
            dependencies,
            buffers: buffers
                .iter()
                .map(|id| id.0)
                .zip(N::buffers().into_iter().map(buffer_state_usage))
                .collect(),
            images: images
                .iter()
                .map(|id| id.0)
                .zip(N::images().into_iter().map(image_state_usage))
                .collect(),
        };
        assert_eq!(pass.buffers.len(), buffers.len());
        assert_eq!(pass.images.len(), images.len());
        pass
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
        buffers: Vec<BufferInfo<B>>,
        images: Vec<ImageInfo<B>>,
        frames: usize,
        family: &B::QueueFamily,
        device: &mut D,
        aux: &mut T,
    ) -> Box<AnyNode<B, D, T>> {
        let node = N::build(
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

    pub(crate) fn build<S, W, U, I>(
        &self,
        submission: &Submission<SyncData<S, W>>,
        image_chains: &ImageChains,
        buffers: &[BufferResource<U>],
        images: &[ImageResource<I>],
        frames: usize,
        family: &B::QueueFamily,
        device: &mut D,
        aux: &mut T,
    ) -> Box<AnyNode<B, D, T>>
    where
        U: Borrow<B::Buffer>,
        I: Borrow<B::Image>,
    {
        let buffers = buffer_info(self.buffers.iter().cloned(), buffers, submission);
        let images = image_info(self.images.iter().cloned(), images, image_chains, submission);

        self.builder
            .build(buffers, images, frames, family, device, aux)
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

fn buffer_info<'a, B, T, U, S, W>(buffers: T, resources: &'a [BufferResource<U>], submission: &Submission<SyncData<S, W>>) -> Vec<BufferInfo<'a, B>>
where
    B: Backend,
    T: Iterator<Item = BufferId>,
    U: Borrow<B::Buffer>,
{
    buffers
        .map(|id| {
            let ref resource = resources[id.0.index() as usize];

            BufferInfo {
                id,
                barriers: Barriers {
                    acquire: {
                        let Range { ref start, ref end } =
                            submission.sync().acquire.buffers[&id.0].states;
                        (start.access, start.stages)..(end.access, end.stages)
                    },
                    release: {
                        let Range { ref start, ref end } =
                            submission.sync().release.buffers[&id.0].states;
                        (start.access, start.stages)..(end.access, end.stages)
                    },
                },
                size: resource.size,
                buffers: resource.buffers.iter().map(Borrow::borrow).collect(),
            }
        })
        .collect()
}

fn image_info<'a, B, T, I, S, W>(
    images: T,
    resources: &'a [ImageResource<I>],
    chains: &ImageChains,
    submission: &Submission<SyncData<S, W>>,
) -> Vec<ImageInfo<'a, B>>
where
    B: Backend,
    T: Iterator<Item = ImageId>,
    I: Borrow<B::Image>,
{
    images
        .map(|id| {
            let ref resource = resources[id.0.index() as usize];

            ImageInfo {
                id,
                barriers: Barriers {
                    acquire: {
                        let Range { ref start, ref end } =
                            submission.sync().acquire.images[&id.0].states;
                        ((start.access, start.layout), start.stages)
                            ..((end.access, end.layout), end.stages)
                    },
                    release: {
                        let Range { ref start, ref end } =
                            submission.sync().release.images[&id.0].states;
                        ((start.access, start.layout), start.stages)
                            ..((end.access, end.layout), end.stages)
                    },
                },
                layout: chains[&id.0].link(submission.image(id.0)).state().layout,
                kind: resource.kind,
                format: resource.format,
                images: resource.images.iter().map(Borrow::borrow).collect(),
            }
        })
        .collect()
}
