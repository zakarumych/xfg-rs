use std::{
    borrow::Borrow, collections::HashMap, iter::once, marker::PhantomData, ops::AddAssign,
    sync::{atomic::AtomicUsize, Arc},
};

use chain::{
    collect::collect, pass::{Pass, PassId, StateUsage}, resource::{Id, State}, schedule::Schedule,
    sync::{sync, SyncData},
};
use either::*;
use hal::{
    buffer, format::Format, image,
    queue::{QueueFamily, QueueFamilyId, RawCommandQueue, RawSubmission}, window::Backbuffer,
    Backend, Device,
};

use smallvec::SmallVec;

use node::{
    build::NodeBuilder, low::{AnyNode, AnyNodeBuilder}, present::{PresentBuilder, PresentNode},
    Node,
};
use util::*;

pub struct Graph<B: Backend, D, T, U, I> {
    nodes: Vec<Box<AnyNode<B, D, T>>>,
    schedule: Schedule<SyncData<usize, usize>>,
    semaphores: Vec<B::Semaphore>,
    buffers: Vec<BufferResource<U>>,
    images: Vec<ImageResource<I>>,
}

impl<B, D, T, U, I> Graph<B, D, T, U, I>
where
    B: Backend,
    D: Device<B>,
{
    /// Perform graph execution.
    /// Run every node of the graph and submit resulting command buffers to the queues.
    ///
    /// # Parameters
    ///
    /// `frame`     - frame index. This index must be less than `frames` specified in `GraphBuilder::build`
    ///               Caller must wait for all `fences` from last time this function was called with same `frame` index.
    ///
    /// `cqueueus`  - function to get `CommandQueue` by `QueueFamilyId` and index.
    ///               `Graph` guarantees that it will submit only command buffers
    ///               allocated from the command pool associated with specified `QueueFamilyId`.
    ///
    /// `device`    - `Device<B>` implementation. `B::Device` or wrapper.
    ///
    /// `aux`       - auxiliary data that `Node`s use.
    ///
    /// `fences`    - vector of fences that will be signaled after all commands are complete.
    ///               Fences that are attached to last submissions of every queue are reset.
    ///               This function may not use all fences. Unused fences are left in signalled state.
    ///               If this function needs more fences they will be allocated from `device` and pushed to this `Vec`.
    ///               So it's OK to start with empty `Vec`.
    pub fn run<'a>(
        &mut self,
        command_queues: &mut HashMap<QueueFamilyId, Vec<B::CommandQueue>>,
        device: &mut D,
        aux: &mut T,
        fences: &mut Vec<B::Fence>,
    ) -> usize {
        let mut fence_index = 0;

        for family in self.schedule.iter() {
            for queue in family.iter() {
                let qid = queue.id();
                let command_queue = command_queues
                    .get_mut(&qid.family())
                    .unwrap()
                    .get_mut(qid.index())
                    .unwrap();
                for (sid, submission) in queue.iter() {
                    let ref mut node = self.nodes[submission.pass().0];

                    let fence = if sid.index() == queue.len() - 1 {
                        while fences.len() <= fence_index {
                            fences.push(device.create_fence(false));
                        }
                        fence_index += 1;
                        device.reset_fence(&fences[fence_index - 1]);
                        Some(&fences[fence_index - 1])
                    } else {
                        None
                    };

                    node.run(
                        submission.sync(),
                        command_queue,
                        &mut self.semaphores,
                        fence,
                        device,
                        aux,
                    );
                }
            }
        }

        fence_index
    }

    /// Dispose of the graph.
    pub fn dispose(self, device: &mut D, aux: &mut T) {
        for node in self.nodes {
            node.dispose(device, aux);
        }
    }
}

pub struct GraphBuilder<B: Backend, D, T, U, I> {
    builders: Vec<Option<Box<AnyNodeBuilder<B, D, T, U, I>>>>,
    buffers: Vec<u64>,
    images: Vec<(image::Kind, Format)>,
}

impl<B, D, T, U, I> GraphBuilder<B, D, T, U, I>
where
    B: Backend,
    D: Device<B>,
    U: Borrow<B::Buffer>,
    I: Borrow<B::Image>,
{
    /// Create new `GraphBuilder`
    pub fn new() -> Self {
        GraphBuilder {
            builders: Vec::new(),
            buffers: Vec::new(),
            images: Vec::new(),
        }
    }

    /// Create new buffer owned by graph.
    pub fn create_buffer(&mut self, size: u64) -> BufferId {
        self.buffers.push(size);
        BufferId::new(self.buffers.len() as u32 - 1)
    }

    /// Create new image owned by graph.
    pub fn create_image(&mut self, kind: image::Kind, format: Format) -> ImageId {
        self.images.push((kind, format));
        ImageId::new(self.images.len() as u32 - 1)
    }

    /// Add node to the graph.
    pub fn add_node<N>(&mut self, builder: NodeBuilder<N>)
    where
        N: Node<B, D, T>,
    {
        self.builders.push(Some(Box::new(builder)));
    }

    /// Build `Graph`.
    ///
    /// # Parameters
    ///
    /// `frames`        - maximum number of frames `Graph` will render simultaneously.
    ///
    /// `families`      - `Iterator` of `B::QueueFamily`s.
    ///
    /// `device`    - `Device<B>` implementation. `B::Device` or wrapper.
    ///
    /// `aux`       - auxiliary data that `Node`s use.
    pub fn build<'a, F, X, Y, P>(
        self,
        families: F,
        mut buffer: X,
        mut image: Y,
        presents: P,
        device: &mut D,
        aux: &mut T,
    ) -> Graph<B, D, T, U, I>
    where
        F: IntoIterator,
        F::Item: Borrow<B::QueueFamily>,
        X: FnMut(u64, buffer::Usage, &mut D, &mut T) -> U,
        Y: FnMut(image::Kind, Format, image::Usage, &mut D, &mut T) -> I,
        P: IntoIterator<Item = PresentBuilder<'a, B>>,
    {
        use chain::{build, pass::Pass};

        let families = families.into_iter().collect::<Vec<_>>();
        let families = families.iter().map(Borrow::borrow).collect::<Vec<_>>();

        let mut builders: Vec<Option<Box<AnyNodeBuilder<B, D, T, U, I> + 'a>>> = self.builders;
        let present_dependencies: Vec<_> = (0..builders.len()).map(PassId).collect();

        for present in presents {
            builders.push(Some(Box::new(
                present.with_dependencies(present_dependencies.clone()),
            )));
        }

        trace!("Schedule nodes execution");
        let mut passes: Vec<Pass> = builders
            .iter()
            .enumerate()
            .map(|(i, b)| b.as_ref().unwrap().pass(PassId(i), &families))
            .collect();

        let chains = collect(passes, |qid| {
            find_family::<B, _>(families.iter().cloned(), qid).max_queues()
        });

        let mut semaphores = GenId::new();
        let schedule = sync(&chains, || {
            let id = semaphores.next();
            (id, id)
        });

        trace!("Allocate buffers");
        let buffers = self
            .buffers
            .iter()
            .enumerate()
            .map(|(index, &size)| {
                let usage = chains.buffers.get(&Id::new(index as u32)).map_or(buffer::Usage::empty(), |chain| chain.usage());
                BufferResource {
                    size,
                    buffer: buffer(size, usage, device, aux),
                }
            })
            .collect::<Vec<_>>();

        trace!("Allocate images");
        let images = self
            .images
            .iter()
            .enumerate()
            .map(|(index, &(kind, format))| {
                let usage = chains.images.get(&Id::new(index as u32)).map_or(image::Usage::empty(), |chain| chain.usage());
                ImageResource {
                    kind,
                    format,
                    image: image(kind, format, usage, device, aux),
                }
            })
            .collect::<Vec<_>>();

        let mut nodes: Vec<Option<Box<AnyNode<B, D, T>>>> =
            (0..builders.len()).map(|_| None).collect();

        for family in schedule.iter() {
            for queue in family.iter() {
                for (sid, submission) in queue.iter() {
                    let node = builders[submission.pass().0].take().unwrap().build(
                        submission,
                        &chains.buffers,
                        &buffers,
                        &chains.images,
                        &images,
                        find_family::<B, _>(families.iter().cloned(), sid.family()),
                        device,
                        aux,
                    );
                    nodes[submission.pass().0] = Some(node);
                }
            }
        }

        Graph {
            nodes: nodes.into_iter().map(Option::unwrap).collect(),
            schedule: schedule,
            semaphores: (0..semaphores.total())
                .map(|_| device.create_semaphore())
                .collect(),
            buffers,
            images,
        }
    }
}

struct GenId<T> {
    next: T,
}

impl<T> GenId<T>
where
    T: Copy + From<u8> + AddAssign,
{
    fn new() -> Self {
        GenId { next: 0.into() }
    }

    fn next(&mut self) -> T {
        let last = self.next;
        self.next += 1u8.into();
        last
    }

    fn total(self) -> T {
        self.next
    }
}

fn find_family<'a, B, F>(families: F, qid: QueueFamilyId) -> &'a B::QueueFamily
where
    B: Backend,
    F: IntoIterator<Item = &'a B::QueueFamily>,
{
    families.into_iter().find(|qf| qf.id() == qid).unwrap()
}
