use {
    chain::{
        chain::{BufferChains, ImageChains}, pass::{Pass, PassId, StateUsage}, resource::State,
        schedule::Submission, sync::SyncData,
    },
    hal::{
        buffer, command::{CommandBufferFlags, ImageCopy, RawCommandBuffer, RawLevel},
        format::{ChannelType, Format}, image, pool::{CommandPoolCreateFlags, RawCommandPool},
        memory::{Barrier, Dependencies},
        pso::PipelineStage, queue::{QueueFamily, RawCommandQueue, RawSubmission},
        window::{Backbuffer, FrameSync, Surface, Swapchain, SwapchainConfig}, Backend, Device,
    },
    node::low::{AnyNode, AnyNodeBuilder}, smallvec::SmallVec,
    std::{borrow::Borrow, collections::HashMap, iter::once, mem::replace}, util::*,
};

pub struct PresentBuilder<'a, B: Backend> {
    pub(crate) id: ImageId,
    pub(crate) format: Format,
    pub(crate) surface: &'a mut B::Surface,
    pub(crate) dependencies: Vec<PassId>,
}

impl<'a, B> PresentBuilder<'a, B>
where
    B: Backend,
{
    /// Create present builder
    pub fn new(id: ImageId, format: Format, surface: &'a mut B::Surface) -> Self {
        PresentBuilder {
            id,
            format,
            surface,
            dependencies: Vec::new(),
        }
    }

    /// Add dependencies to the present builder.
    pub fn with_dependencies<I>(mut self, deps: I) -> Self
    where
        I: IntoIterator<Item = PassId>,
    {
        self.dependencies.extend(deps);
        self
    }

    fn build_node<I, D>(
        self,
        submission: &Submission<SyncData<usize, usize>>,
        chains: &ImageChains,
        resources: &[ImageResource<I>],
        family: &B::QueueFamily,
        device: &mut D,
    ) -> PresentNode<B>
    where
        D: Device<B>,
        I: Borrow<B::Image>,
    {
        let (swapchain, backbuffer) = device.create_swapchain(
            self.surface,
            SwapchainConfig {
                color_format: self.format,
                depth_stencil_format: None,
                image_count: 3,
                image_usage: image::Usage::TRANSFER_DST,
            },
        );

        let mut pool = device.create_command_pool(family.id(), CommandPoolCreateFlags::empty());

        let ref chain = chains[&self.id.0];
        let link = chain.link(submission.image(self.id.0));
        let ref resource = resources[self.id.index()];

        let ref acquire = submission.sync().acquire.images.get(&self.id.0);
        let ref release = submission.sync().release.images.get(&self.id.0);

        let per_frame = match backbuffer {
            Backbuffer::Images(ref images) => {
                let cbufs = pool.allocate(images.len(), RawLevel::Primary);
                cbufs
                    .into_iter()
                    .enumerate()
                    .map(|(index, mut cbuf)| {
                        cbuf.begin(CommandBufferFlags::EMPTY, Default::default());
                        acquire.map(|acquire| {
                            cbuf.pipeline_barrier(
                                acquire.states.start.stages .. acquire.states.end.stages,
                                Dependencies::empty(),
                                Some(Barrier::Image {
                                    states: (acquire.states.start.access, acquire.states.start.layout) .. (acquire.states.end.access, acquire.states.end.layout),
                                    target: resource.image.borrow(),
                                    range: image::SubresourceRange {
                                        aspects: resource.format.aspects(),
                                        levels: 0..1,
                                        layers: 0..1,
                                    }
                                })
                            );
                        });
                        cbuf.copy_image(
                            resource.image.borrow(),
                            link.state().layout,
                            &images[index],
                            image::Layout::TransferDstOptimal,
                            Some(ImageCopy {
                                src_subresource: image::SubresourceLayers {
                                    aspects: resource.format.aspects(),
                                    level: 0,
                                    layers: 0..1,
                                },
                                src_offset: image::Offset { x: 0, y: 0, z: 0 },
                                dst_subresource: image::SubresourceLayers {
                                    aspects: self.format.aspects(),
                                    level: 0,
                                    layers: 0..1,
                                },
                                dst_offset: image::Offset { x: 0, y: 0, z: 0 },
                                extent: resource.kind.extent(),
                            }),
                        );
                        release.map(|release| {
                            cbuf.pipeline_barrier(
                                release.states.start.stages .. release.states.end.stages,
                                Dependencies::empty(),
                                Some(Barrier::Image {
                                    states: (release.states.start.access, release.states.start.layout) .. (release.states.end.access, release.states.end.layout),
                                    target: resource.image.borrow(),
                                    range: image::SubresourceRange {
                                        aspects: resource.format.aspects(),
                                        levels: 0..1,
                                        layers: 0..1,
                                    }
                                })
                            );
                        });
                        cbuf.pipeline_barrier(
                            PipelineStage::TRANSFER .. PipelineStage::TOP_OF_PIPE,
                            Dependencies::empty(),
                            Some(Barrier::Image {
                                states: (image::Access::TRANSFER_READ, image::Layout::TransferDstOptimal) .. (image::Access::all(), image::Layout::Present),
                                target: &images[index],
                                range: image::SubresourceRange {
                                    aspects: self.format.aspects(),
                                    levels: 0..1,
                                    layers: 0..1,
                                }
                            })
                        );
                        cbuf.finish();

                        (device.create_semaphore(), device.create_semaphore(), cbuf)
                    })
                    .collect()
            }
            Backbuffer::Framebuffer(_) => unimplemented!(),
        };

        PresentNode {
            per_frame,
            free: Some(device.create_semaphore()),
            swapchain: swapchain,
            backbuffer: backbuffer,
            pool,
        }
    }
}

impl<'a, B, D, T, U, I> AnyNodeBuilder<B, D, T, U, I> for PresentBuilder<'a, B>
where
    B: Backend,
    D: Device<B>,
    U: Borrow<B::Buffer>,
    I: Borrow<B::Image>,
{
    fn name(&self) -> &str {
        "PresentNode"
    }

    fn pass(&self, id: PassId, families: &[&B::QueueFamily]) -> Pass {
        Pass {
            id,
            family: families
                .iter()
                .find(|qf| self.surface.supports_queue_family(qf))
                .unwrap()
                .id(),
            queue: None,
            dependencies: Vec::new(),
            buffers: HashMap::new(),
            images: once((
                self.id.0,
                StateUsage {
                    state: State {
                        access: image::Access::TRANSFER_READ,
                        layout: image::Layout::TransferSrcOptimal,
                        stages: PipelineStage::TRANSFER,
                    },
                    usage: image::Usage::TRANSFER_SRC,
                },
            )).collect(),
        }
    }

    fn build(
        self: Box<Self>,
        submission: &Submission<SyncData<usize, usize>>,
        _: &BufferChains,
        _: &[BufferResource<U>],
        chains: &ImageChains,
        resources: &[ImageResource<I>],
        family: &B::QueueFamily,
        device: &mut D,
        _aux: &mut T,
    ) -> Box<AnyNode<B, D, T>> {
        Box::new(self.build_node(submission, chains, resources, family, device))
    }
}

pub struct PresentNode<B: Backend> {
    per_frame: Vec<(B::Semaphore, B::Semaphore, B::CommandBuffer)>,
    free: Option<B::Semaphore>,
    swapchain: B::Swapchain,
    backbuffer: Backbuffer<B>,
    pool: B::CommandPool,
}

impl<B, D, T> AnyNode<B, D, T> for PresentNode<B>
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
        _device: &mut D,
        _aux: &'a T,
    ) {
        assert!(sync.acquire.signal.is_empty());
        assert!(sync.release.signal.is_empty());
        assert!(sync.release.wait.is_empty());

        let wait = sync
            .acquire
            .wait
            .iter()
            .map(|wait| (&semaphores[*wait.semaphore()], wait.stage()));
        let acquire = self.free.take().unwrap();

        let frame = self
            .swapchain
            .acquire_frame(FrameSync::Semaphore(&acquire))
            .id();

        self.free = Some(replace(&mut self.per_frame[frame].0, acquire));
        let (ref acquire, ref release, ref cbuf) = self.per_frame[frame];

        let submission = RawSubmission {
            wait_semaphores: &wait
                .chain(Some((acquire, PipelineStage::TRANSFER)))
                .collect::<SmallVec<[_; 8]>>(),
            signal_semaphores: &[release],
            cmd_buffers: Some(cbuf),
        };

        unsafe {
            queue.submit_raw(submission, fence);
            queue.present(Some(&mut self.swapchain), Some(release));
        }
    }

    fn dispose(self: Box<Self>, device: &mut D, aux: &mut T) {
        unimplemented!()
    }
}
