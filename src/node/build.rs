use std::{borrow::Borrow, cmp::Ordering, marker::PhantomData, ops::Range};

use hal::{
    buffer, format::Format, image, pool::{CommandPool, CommandPoolCreateFlags}, pso::PipelineStage,
    queue::{Capability, CommandQueue, QueueFamily, QueueFamilyId, QueueType}, window::Backbuffer,
    Backend, Device,
};

use chain::{
    chain::{BufferChains, ImageChains}, pass::{Pass, PassId, StateUsage},
    resource::{Access, Buffer, BufferLayout, Image, State}, schedule::Submission, sync::SyncData,
};

use node::{low::*, Barriers, BufferInfo, ImageInfo, Node, NodeDesc};
use util::*;

pub struct NodeBuilder<N> {
    buffers: Vec<BufferId>,
    images: Vec<ImageId>,
    dependencies: Vec<PassId>,
    _pd: PhantomData<N>,
}

impl<N> NodeBuilder<N>
where
    N: NodeDesc,
{
    /// Create `NodeBuilder` that builds node of type `N`.
    pub fn new() -> Self {
        NodeBuilder {
            buffers: Vec::new(),
            images: Vec::new(),
            dependencies: Vec::new(),
            _pd: PhantomData,
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

    /// Add dependency to another node.
    pub fn add_dependency(&mut self, id: NodeId) -> &mut Self {
        self.dependencies.push(id.0);
        self
    }

    /// Add dependency to another node.
    pub fn with_dependency(mut self, id: NodeId) -> Self {
        self.add_dependency(id);
        self
    }
}

impl<B, D, T, U, I, N> AnyNodeBuilder<B, D, T, U, I> for NodeBuilder<N>
where
    B: Backend,
    D: Device<B>,
    N: Node<B, D, T>,
    I: Borrow<B::Image>,
    U: Borrow<B::Buffer>,
{
    fn name(&self) -> &str {
        N::name()
    }

    fn pass(&self, id: PassId, families: &[&B::QueueFamily]) -> Pass {
        let pass = Pass {
            id,
            family: pick_queue_family::<B, N::Capability, _>(families.iter().cloned()),
            queue: None,
            dependencies: self.dependencies.clone(),
            buffers: self
                .buffers
                .iter()
                .map(|id| id.0)
                .zip(N::buffers().into_iter().map(buffer_state_usage))
                .collect(),
            images: self
                .images
                .iter()
                .map(|id| id.0)
                .zip(N::images().into_iter().map(image_state_usage))
                .collect(),
        };
        assert_eq!(pass.buffers.len(), self.buffers.len());
        assert_eq!(pass.images.len(), self.images.len());
        pass
    }

    fn build(
        self: Box<Self>,
        submission: &Submission<SyncData<usize, usize>>,
        _buffer_chains: &BufferChains,
        buffers: &[BufferResource<U>],
        image_chains: &ImageChains,
        images: &[ImageResource<I>],
        family: &B::QueueFamily,
        device: &mut D,
        aux: &mut T,
    ) -> Box<AnyNode<B, D, T>> {
        let buffer_info = buffer_info(&self.buffers, &*buffers, submission);
        let image_info = image_info(&self.images, &*images, image_chains, submission);

        let pools = |device: &mut _, flags| create_typed_pool(family, flags, device);
        let node = N::build(buffer_info, image_info, pools, device, aux);
        Box::new((node,))
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

fn buffer_info<'a, U, S, W>(
    buffers: &[BufferId],
    resources: &'a [BufferResource<U>],
    submission: &Submission<SyncData<S, W>>,
) -> Vec<BufferInfo<'a, U>> {
    buffers
        .iter()
        .map(|&id| {
            let ref resource = resources[id.0.index() as usize];
            BufferInfo {
                id,
                barriers: Barriers {
                    acquire: submission.sync().acquire.buffers.get(&id.0).map(|barrier| {
                        let Range { ref start, ref end } = barrier.states;
                        (start.access, start.stages)..(end.access, end.stages)
                    }),
                    release: submission.sync().release.buffers.get(&id.0).map(|barrier| {
                        let Range { ref start, ref end } = barrier.states;
                        (start.access, start.stages)..(end.access, end.stages)
                    }),
                },
                size: resource.size,
                buffer: &resource.buffer,
            }
        })
        .collect()
}

fn image_info<'a, I, S, W>(
    images: &[ImageId],
    resources: &'a [ImageResource<I>],
    chains: &ImageChains,
    submission: &Submission<SyncData<S, W>>,
) -> Vec<ImageInfo<'a, I>> {
    images
        .iter()
        .map(|&id| {
            let ref resource = resources[id.0.index() as usize];
            let link = chains[&id.0].link(submission.image(id.0));
            ImageInfo {
                id,
                barriers: Barriers {
                    acquire: submission.sync().acquire.images.get(&id.0).map(|barrier| {
                        let Range { ref start, ref end } = barrier.states;
                        ((start.access, start.layout), start.stages)
                            ..((end.access, end.layout), end.stages)
                    }),
                    release: submission.sync().release.images.get(&id.0).map(|barrier| {
                        let Range { ref start, ref end } = barrier.states;
                        ((start.access, start.layout), start.stages)
                            ..((end.access, end.layout), end.stages)
                    }),
                },
                layout: link.state().layout,
                kind: resource.kind,
                format: resource.format,
                clear: resource.clear.and_then(|clear| {
                    if submission.image(id.0) == 0 {
                        assert!(link.state().access.is_write(), "First node must do writing in order to be able to clear image");
                        Some(clear)
                    } else {
                        None
                    }
                }),
                image: &resource.image,
            }
        })
        .collect()
}
