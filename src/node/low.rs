use std::{borrow::Borrow, cmp::Ordering, marker::PhantomData, ops::Range};

use hal::{
    buffer, format::Format, image, pool::{CommandPool, CommandPoolCreateFlags}, pso::PipelineStage,
    queue::{Capability, CommandQueue, QueueFamily, QueueFamilyId, QueueType}, window::Backbuffer,
    Backend, Device,
};

use chain::{
    chain::{BufferChains, ImageChains}, pass::{Pass, PassId, StateUsage},
    resource::{Buffer, BufferLayout, Image, State}, schedule::Submission, sync::SyncData,
};

use node::{Barriers, BufferInfo, ImageInfo, Node, NodeDesc};
use util::*;

pub trait AnyNodeBuilder<B, D, T, U, I>: Send + Sync
where
    B: Backend,
    U: Borrow<B::Buffer>,
    I: Borrow<B::Image>,
{
    fn name(&self) -> &str;

    fn pass(&self, id: PassId, families: &[&B::QueueFamily]) -> Pass;

    fn build(
        self: Box<Self>,
        submission: &Submission<SyncData<usize, usize>>,
        buffer_chains: &BufferChains,
        buffers: &[BufferResource<U>],
        image_chains: &ImageChains,
        images: &[ImageResource<I>],
        family: &B::QueueFamily,
        device: &mut D,
        aux: &mut T,
    ) -> Box<AnyNode<B, D, T>>;
}

pub trait AnyNode<B, D, T>: Send + Sync
where
    B: Backend,
    D: Device<B>,
{
    fn run<'a>(
        &'a mut self,
        sync: &SyncData<usize, usize>,
        queue: &mut B::CommandQueue,
        semaphores: &mut [B::Semaphore],
        fence: Option<&B::Fence>,
        device: &mut D,
        aux: &'a T,
    );

    fn dispose(self: Box<Self>, device: &mut D, aux: &mut T);
}

impl<B, D, T, N> AnyNode<B, D, T> for (N,)
where
    B: Backend,
    D: Device<B>,
    N: Node<B, D, T>,
{
    fn run<'a>(
        &'a mut self,
        sync: &SyncData<usize, usize>,
        queue: &mut B::CommandQueue,
        semaphores: &mut [B::Semaphore],
        fence: Option<&B::Fence>,
        device: &mut D,
        aux: &'a T,
    ) {
        let queue: &mut CommandQueue<B, N::Capability> = unsafe { ::std::mem::transmute(queue) };

        assert!(sync.acquire.signal.is_empty());
        assert!(sync.release.wait.is_empty());

        let wait = sync
            .acquire
            .wait
            .iter()
            .map(|wait| (&semaphores[*wait.semaphore()], wait.stage()));
        let signal = sync
            .release
            .signal
            .iter()
            .map(|signal| &semaphores[*signal.semaphore()]);

        N::run(&mut self.0, wait, queue, signal, fence, device, aux)
    }

    fn dispose(self: Box<Self>, device: &mut D, aux: &mut T) {
        self.0.dispose(device, aux);
    }
}
