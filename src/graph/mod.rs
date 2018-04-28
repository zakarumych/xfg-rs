
use std::{marker::PhantomData, ops::AddAssign};
use chain::{collect::Chains, schedule::Schedule, sync::Sync};
use hal::{Backend, Device,
          queue::{QueueFamily, QueueFamilyId}
          };
use node::{Info, AnyNode, Node};

struct NextId<T> {
    next: T,
}

impl<T> NextId<T>
where
    T: Copy + From<u8> + AddAssign,
{
    fn new() -> Self {
        Self::default()
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

impl<T> Default for NextId<T>
where
    T: From<u8>,
{
    fn default() -> Self {
        NextId {
            next: 0u8.into(),
        }
    }
}

pub struct Graph<B: Backend, D, T> {
    nodes: Vec<Box<AnyNode<B, D, T>>>,
    schedule: Schedule<Sync<usize, usize>>,
    semaphores: Vec<B::Semaphore>,
}

pub struct GraphBuilder<B: Backend, D, T> {
    builders: Vec<Info<B, D, T>>,
    next_buffer_id: NextId<u32>,
    next_image_id: NextId<u32>,
}

impl<B, D, T> GraphBuilder<B, D, T>
where
    B: Backend,
    D: Device<B>,
{
    pub fn add_node<N>(&mut self)
    where
        N: Node<B, D, T>
    {
        self.builders.push(Info::new::<N>());
    }

    pub fn build(&self, frames: usize, families: &[B::QueueFamily], device: &mut D, aux: &mut T) -> Graph<B, D, T> {
        use chain::{pass::Pass, build};

        let mut semaphores = NextId::new();

        let passes: Vec<Pass> = self.builders.iter().enumerate().map(|(i, b)| b.chain(i, families)).collect();
        let chains = build(passes, |qid| find_family::<B>(families, qid).max_queues(), || {
            let id = semaphores.next();
            (id, id)
        });

        let nodes: Vec<Option<Box<AnyNode<B, D, T>>>> = (0..self.builders.len()).map(|_| None).collect();

        // let nodes = self.builders.iter().zip(&passes).map(|(b, p)| b.build(frames, find_family::<B>(families, p.family), device, aux)).collect();

        Graph {
            nodes: nodes.into_iter().map(Option::unwrap).collect(),
            schedule: chains.schedule,
            semaphores: (0..semaphores.total()).map(|_| device.create_semaphore()).collect(),
        }
    }
}

fn find_family<B>(families: &[B::QueueFamily], qid: QueueFamilyId) -> &B::QueueFamily
where
    B: Backend,
{
    families.iter().find(|qf| qf.id() == qid).unwrap()
}
