use chain::{resource::Id, schedule::Schedule, sync::SyncData};
use hal::{Backend, Device, queue::{QueueFamily, QueueFamilyId, RawCommandQueue, RawSubmission}, image, buffer, format::Format};
use std::{borrow::Borrow, ops::AddAssign};

use smallvec::SmallVec;

use id::{BufferId, ImageId};
use node::{Node, build::{AnyNode, NodeBuilder}};

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
    pub fn run<'a, C>(
        &mut self,
        frame: usize,
        mut cqueues: C,
        device: &mut D,
        aux: &mut T,
        fences: &mut Vec<B::Fence>,
    ) where
        C: FnMut(QueueFamilyId, usize) -> &'a mut B::CommandQueue,
    {
        let mut fence_index = 0;

        let ref semaphores = self.semaphores;
        for family in self.schedule.iter() {
            for queue in family.iter() {
                let qid = queue.id();
                let cqueue = cqueues(qid.family(), qid.index());
                for (sid, submission) in queue.iter() {
                    let ref mut node = self.nodes[submission.pass().0];

                    assert!(
                        submission.sync().acquire.signal.is_empty()
                            && submission.sync().release.wait.is_empty()
                    );

                    let wait = submission
                        .sync()
                        .acquire
                        .wait
                        .iter()
                        .map(|wait| (&semaphores[*wait.semaphore()], wait.stage()))
                        .collect::<SmallVec<[_; 16]>>();
                    let signal = submission
                        .sync()
                        .release
                        .signal
                        .iter()
                        .map(|signal| &semaphores[*signal.semaphore()])
                        .collect::<SmallVec<[_; 16]>>();

                    let mut cbufs = SmallVec::new();
                    node.run(frame, device, aux, &mut cbufs);

                    let raw_submission = RawSubmission {
                        wait_semaphores: &wait,
                        signal_semaphores: &signal,
                        cmd_buffers: cbufs,
                    };

                    let fence = if sid.index() == queue.len() - 1 {
                        while fences.len() <= fence_index {
                            fences.push(device.create_fence(false));
                        }
                        fence_index += 1;
                        Some(&fences[fence_index - 1])
                    } else {
                        None
                    };

                    unsafe {
                        cqueue.submit_raw(raw_submission, fence);
                    }
                }
            }
        }
    }
}


pub struct GraphBuilder<B: Backend, D, T> {
    builders: Vec<NodeBuilder<B, D, T>>,
    buffers: Vec<u64>,
    images: Vec<(image::Kind, Format)>,
}

impl<B, D, T> GraphBuilder<B, D, T>
where
    B: Backend,
    D: Device<B>,
{
    /// Allocate new buffer id.
    pub fn create_buffer(&mut self, size: u64) -> BufferId {
        self.buffers.push(size);
        BufferId(Id::new(self.buffers.len() as u32 - 1))
    }

    /// Allocate new image id.
    pub fn create_image(&mut self, kind: image::Kind, format: Format) -> ImageId {
        self.images.push((kind, format));
        ImageId(Id::new(self.images.len() as u32 - 1))
    }

    /// Add node to the graph.
    pub fn add_node<N>(&mut self, node: NodeBuilder<B, D, T>)
    where
        N: Node<B, D, T>,
    {
        self.builders.push(node);
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
    pub fn build<F, X, Y, U, I>(
        &self,
        frames: usize,
        families: F,
        mut buffer: X,
        mut image: Y,
        device: &mut D,
        aux: &mut T,
    ) -> Graph<B, D, T, U, I>
    where
        F: IntoIterator,
        F::Item: Borrow<B::QueueFamily>,
        U: Borrow<B::Buffer>,
        I: Borrow<B::Image>,
        X: FnMut(u64, buffer::Usage, &mut D, &mut T) -> U,
        Y: FnMut(image::Kind, Format, image::Usage, &mut D, &mut T) -> I,
    {
        use chain::{build, pass::Pass};

        let families = families.into_iter().collect::<Vec<_>>();
        let families = families.iter().map(Borrow::borrow).collect::<Vec<_>>();

        trace!("Schedule nodes execution");
        let mut semaphores = GenId::new();

        let passes: Vec<Pass> = self.builders
            .iter()
            .enumerate()
            .map(|(i, b)| b.chain(i, &families))
            .collect();
        let chains = build(
            passes,
            |qid| find_family::<B, _>(families.iter().cloned(), qid).max_queues(),
            || {
                let id = semaphores.next();
                (id, id)
            },
        );

        trace!("Allocate buffers");
        let buffers = self.buffers.iter().enumerate().map(|(index, &size)| {
            let usage = chains.buffers[&Id::new(index as u32)].usage();
            BufferResource {
                size,
                buffers: (0 .. frames).map(|_| buffer(size, usage, device, aux)).collect()
            }
        }).collect::<Vec<_>>();

        trace!("Allocate images");
        let images = self.images.iter().enumerate().map(|(index, &(kind, format))| {
            let usage = chains.images[&Id::new(index as u32)].usage();
            ImageResource {
                kind,
                format,
                images: (0 .. frames).map(|_| image(kind, format, usage, device, aux)).collect()
            }
        }).collect::<Vec<_>>();

        let mut nodes: Vec<Option<Box<AnyNode<B, D, T>>>> =
            (0..self.builders.len()).map(|_| None).collect();

        for family in chains.schedule.iter() {
            for queue in family.iter() {
                for (sid, submission) in queue.iter() {
                    let node = self.builders[submission.pass().0].build(
                        submission,
                        &chains.images,
                        &buffers,
                        &images,
                        frames,
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
            schedule: chains.schedule,
            semaphores: (0..semaphores.total())
                .map(|_| device.create_semaphore())
                .collect(),
            buffers,
            images,
        }
    }
}

pub struct BufferResource<U> {
    pub size: u64,
    pub buffers: Vec<U>,
}

pub struct ImageResource<I> {
    pub kind: image::Kind,
    pub format: Format,
    pub images: Vec<I>,
}

struct GenId<T> {
    next: T,
}

impl<T> GenId<T>
where
    T: Copy + From<u8> + AddAssign,
{
    fn new() -> Self {
        GenId {
            next: 0.into(),
        }
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
