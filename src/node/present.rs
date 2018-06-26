use {
    chain::{
        chain::{BufferChains, ImageChains}, pass::{Pass, PassId, StateUsage}, resource::State,
        schedule::Submission, sync::SyncData,
    },
    hal::{
        buffer,
        command::{
            ClearColorRaw, ClearDepthStencilRaw, CommandBufferFlags, ImageCopy, RawCommandBuffer,
            RawLevel,
        },
        format::{ChannelType, Format}, image, memory::{Barrier, Dependencies},
        pool::{CommandPoolCreateFlags, RawCommandPool}, pso::PipelineStage,
        queue::{QueueFamily, RawCommandQueue, RawSubmission},
        window::{
            Backbuffer, FrameSync, PresentMode, Surface, SurfaceCapabilities, Swapchain,
            SwapchainConfig,
        },
        Backend, Device,
    },
    node::low::{AnyNode, AnyNodeBuilder}, smallvec::SmallVec,
    std::{borrow::Borrow, collections::HashMap, iter::once, mem::replace}, util::*,
};

pub struct PresentBuilder<'a, B: Backend> {
    pub(crate) id: ImageId,
    pub(crate) format: Format,
    pub(crate) surface: &'a mut B::Surface,
    pub(crate) capabilities: SurfaceCapabilities,
    pub(crate) dependencies: Vec<PassId>,
}

impl<'a, B> PresentBuilder<'a, B>
where
    B: Backend,
{
    /// Create present builder
    pub fn new(
        id: ImageId,
        format: Format,
        surface: &'a mut B::Surface,
        capabilities: SurfaceCapabilities,
    ) -> Self {
        PresentBuilder {
            id,
            format,
            surface,
            capabilities,
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
        let extent = self.surface.kind().extent().into();
        let (swapchain, backbuffer) = device.create_swapchain(
            self.surface,
            SwapchainConfig {
                present_mode: PresentMode::Fifo,
                color_format: self.format,
                depth_stencil_format: None,
                image_count: if self.capabilities.image_count.start <= 3 {
                    if self.capabilities.image_count.end >= 3 {
                        3
                    } else {
                        self.capabilities.image_count.end
                    }
                } else {
                    self.capabilities.image_count.start
                },
                image_usage: image::Usage::TRANSFER_DST,
            },
            None,
            &extent,
        );

        let mut pool = device.create_command_pool(family.id(), CommandPoolCreateFlags::empty());

        let ref chain = chains[&self.id.0];
        let link = chain.link(submission.image(self.id.0));
        let ref resource = resources[self.id.0.index() as usize];

        let ref acquire = submission.sync().acquire.images.get(&self.id.0);
        let ref release = submission.sync().release.images.get(&self.id.0);

        let per_frame = match backbuffer {
            Backbuffer::Images(ref backbuffer_images) => {
                let cbufs = pool.allocate(backbuffer_images.len(), RawLevel::Primary);
                cbufs
                    .into_iter()
                    .enumerate()
                    .map(|(index, mut cbuf)| {
                        let ref backbuffer_image = backbuffer_images[index];
                        cbuf.begin(CommandBufferFlags::EMPTY, Default::default());
                        acquire.map(|acquire| {
                            cbuf.pipeline_barrier(
                                acquire.states.start.stages..acquire.states.end.stages,
                                Dependencies::empty(),
                                Some(Barrier::Image {
                                    states: (
                                        acquire.states.start.access,
                                        acquire.states.start.layout,
                                    )
                                        ..(acquire.states.end.access, acquire.states.end.layout),
                                    target: resource.image.borrow(),
                                    range: image::SubresourceRange {
                                        aspects: resource.format.surface_desc().aspects,
                                        levels: 0..1,
                                        layers: 0..1,
                                    },
                                }),
                            );
                        });
                        cbuf.pipeline_barrier(
                            PipelineStage::BOTTOM_OF_PIPE..PipelineStage::TRANSFER,
                            Dependencies::empty(),
                            Some(Barrier::Image {
                                states: (image::Access::empty(), image::Layout::Present)
                                    ..(
                                        image::Access::TRANSFER_READ,
                                        image::Layout::TransferDstOptimal,
                                    ),
                                target: backbuffer_image,
                                range: image::SubresourceRange {
                                    aspects: self.format.surface_desc().aspects,
                                    levels: 0..1,
                                    layers: 0..1,
                                },
                            }),
                        );
                        cbuf.copy_image(
                            resource.image.borrow(),
                            link.state().layout,
                            backbuffer_image,
                            image::Layout::TransferDstOptimal,
                            Some(ImageCopy {
                                src_subresource: image::SubresourceLayers {
                                    aspects: resource.format.surface_desc().aspects,
                                    level: 0,
                                    layers: 0..1,
                                },
                                src_offset: image::Offset { x: 0, y: 0, z: 0 },
                                dst_subresource: image::SubresourceLayers {
                                    aspects: self.format.surface_desc().aspects,
                                    level: 0,
                                    layers: 0..1,
                                },
                                dst_offset: image::Offset { x: 0, y: 0, z: 0 },
                                extent: resource.kind.extent(),
                            }),
                        );
                        cbuf.pipeline_barrier(
                            PipelineStage::TRANSFER..PipelineStage::TOP_OF_PIPE,
                            Dependencies::empty(),
                            Some(Barrier::Image {
                                states: (
                                    image::Access::TRANSFER_READ,
                                    image::Layout::TransferDstOptimal,
                                )
                                    ..(image::Access::empty(), image::Layout::Present),
                                target: backbuffer_image,
                                range: image::SubresourceRange {
                                    aspects: self.format.surface_desc().aspects,
                                    levels: 0..1,
                                    layers: 0..1,
                                },
                            }),
                        );
                        release.map(|release| {
                            cbuf.pipeline_barrier(
                                release.states.start.stages..release.states.end.stages,
                                Dependencies::empty(),
                                Some(Barrier::Image {
                                    states: (
                                        release.states.start.access,
                                        release.states.start.layout,
                                    )
                                        ..(release.states.end.access, release.states.end.layout),
                                    target: resource.image.borrow(),
                                    range: image::SubresourceRange {
                                        aspects: resource.format.surface_desc().aspects,
                                        levels: 0..1,
                                        layers: 0..1,
                                    },
                                }),
                            );
                        });

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
            dependencies: self.dependencies.clone(),
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
        profile!("Present::run");
        assert!(sync.acquire.signal.is_empty());
        assert!(sync.release.signal.is_empty());
        assert!(sync.release.wait.is_empty());

        let wait = sync
            .acquire
            .wait
            .iter()
            .map(|wait| (&semaphores[*wait.semaphore()], wait.stage()));
        let acquire = self.free.take().unwrap();

        let frame = {
            profile!("Acquire frame");
            self.swapchain
                .acquire_image(FrameSync::Semaphore(&acquire))
                .unwrap()
        };

        self.free = Some(replace(&mut self.per_frame[frame as usize].0, acquire));
        let (ref acquire, ref release, ref cbuf) = self.per_frame[frame as usize];

        let submission = RawSubmission {
            wait_semaphores: &wait
                .chain(Some((acquire, PipelineStage::TRANSFER)))
                .collect::<SmallVec<[_; 8]>>(),
            signal_semaphores: &[release],
            cmd_buffers: Some(cbuf),
        };

        trace!("Presenting");
        unsafe {
            profile!("Submit");
            queue.submit_raw(submission, fence);
            queue
                .present(Some((&mut self.swapchain, frame)), Some(release))
                .unwrap();
        }
    }

    fn dispose(self: Box<Self>, _device: &mut D, _aux: &mut T) {
        unimplemented!()
    }
}
