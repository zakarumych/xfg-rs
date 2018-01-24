//! Defines `Pass` trait - main building block for your `Graph`s

mod build;

pub use self::build::PassBuilder;

use std::fmt::Debug;

use gfx_hal::{Backend, Device};
use gfx_hal::command::{CommandBuffer, RenderPassInlineEncoder, ClearValue, Rect};
use gfx_hal::device::ShaderError;
use gfx_hal::format::Format;
use gfx_hal::pso::{DescriptorSetLayoutBinding, ElemStride, Element, GraphicsShaderSet, PipelineStage};
use gfx_hal::queue::capability::{Supports, Transfer, Graphics};

use smallvec::SmallVec;

use bindings::{Binder, BindingsList, Layout};
use descriptors::DescriptorPool;
use frame::{SuperFramebuffer, SuperFrame, pick};

/// Bulding block for `Graph`.
/// Pass is similar to the concept of `gfx_hal::Backend::RenderPass`.
/// But several `Pass`'es may be mapped into single `RenderPass` as subpasses.
/// Pass defines it's inputs and outputs. Also vertex format and bindings.
/// Drawing methods define how to pick data for drawing and record drawing commands.
/// `T` is used by `Pass` implementaions however they like.
/// Note that pass has exclusive access to `T` during preparaion phase.
/// Which executed secuentially and expected to be fast.
/// During drawing phase `T` is shared as passes record drawing commands in parallel.
pub trait Pass<B, T>: Debug
where
    B: Backend,
{
    /// Name of the pass
    const NAME: &'static str;

    /// Input attachments desired format.
    /// Format of actual attachment may be different.
    /// It may be larger if another consumer expects larger format.
    /// It may be smaller because of hardware limitations.
    const INPUTS: usize;

    /// Number of colors to write
    const COLORS: usize;

    /// Does pass writes into depth buffer?
    const DEPTH: bool;

    /// Does pass uses stencil?
    const STENCIL: bool;

    /// Vertices format
    const VERTICES: &'static [(&'static [Element<Format>], ElemStride)];

    type Bindings: BindingsList;

    /// Fill layout with bindings
    fn layout(Layout<()>) -> Layout<Self::Bindings>;

    /// Build render pass
    fn build() -> PassBuilder<'static, B, T>
    where
        Self: 'static + Default,
    {
        PassBuilder::new(Self::default())
    }

    /// Load shaders
    ///
    /// This function gets called during `Graph` build process.
    fn shaders<'a>(
        shaders: &'a mut SmallVec<[B::ShaderModule; 5]>,
        device: &B::Device,
    ) -> Result<GraphicsShaderSet<'a, B>, ShaderError>;

    /// Make preparation for actual drawing commands.
    /// Bind buffers. Transfer data.
    fn prepare<'a, C>(
        &mut self,
        pool: &mut DescriptorPool<B>,
        cbuf: &mut CommandBuffer<B, C>,
        device: &B::Device,
        aux: &mut T,
    ) where
        C: Supports<Transfer>;

    /// Record actual drawing commands in inline fashion.
    fn draw_inline<'a>(
        &mut self,
        binder: Binder<B, Self::Bindings>,
        encoder: RenderPassInlineEncoder<B>,
        device: &B::Device,
        aux: &T,
    );

    /// Cleanup before dropping this pass
    fn cleanup(&mut self, _pool: &mut DescriptorPool<B>, _device: &B::Device, _aux: &mut T);
}

/// Object-safe trait that mirrors `Pass` trait.
/// It's implemented for any type that implements `Pass`.
pub trait AnyPass<B, T>: Debug
where
    B: Backend,
{
    /// Name of the pass
    fn name(&self) -> &'static str;

    /// Input attachments format
    fn inputs(&self) -> usize;

    /// Colors count
    fn colors(&self) -> usize;

    /// Uses depth?
    fn depth(&self) -> bool;

    /// Uses stencil?
    fn stencil(&self) -> bool;

    /// Bindings
    fn bindings(&self) -> SmallVec<[DescriptorSetLayoutBinding; 64]>;

    /// Vertices format
    fn vertices(&self) -> &'static [(&'static [Element<Format>], ElemStride)];

    /// Load shaders
    fn shaders<'a>(
        &self,
        shaders: &'a mut SmallVec<[B::ShaderModule; 5]>,
        device: &B::Device,
    ) -> Result<GraphicsShaderSet<'a, B>, ShaderError>;

    /// Reflects [`Pass::prepare`] function
    ///
    /// [`Pass::prepare`]: trait.Pass.html#tymethod.prepare
    fn prepare(
        &mut self,
        pool: &mut DescriptorPool<B>,
        cbuf: &mut CommandBuffer<B, Transfer>,
        device: &B::Device,
        aux: &mut T,
    );

    /// Reflects [`Pass::draw_inline`] function
    ///
    /// [`Pass::draw_inline`]: trait.Pass.html#tymethod.draw_inline
    fn draw_inline(
        &mut self,
        layout: &B::PipelineLayout,
        encoder: RenderPassInlineEncoder<B>,
        device: &B::Device,
        aux: &T,
    );

    fn cleanup(&mut self, pool: &mut DescriptorPool<B>, device: &B::Device, aux: &mut T);
}

impl<P, B, T> AnyPass<B, T> for P
where
    P: Pass<B, T> + 'static,
    B: Backend,
{
    /// Name of the pass
    fn name(&self) -> &'static str {
        P::NAME
    }

    /// Input attachments format
    fn inputs(&self) -> usize {
        P::INPUTS
    }

    /// Colors count
    fn colors(&self) -> usize {
        P::COLORS
    }

    /// Uses depth?
    fn depth(&self) -> bool {
        P::DEPTH
    }

    /// Uses stencil?
    fn stencil(&self) -> bool {
        P::STENCIL
    }

    /// Bindings
    fn bindings(&self) -> SmallVec<[DescriptorSetLayoutBinding; 64]> {
        Self::layout(Layout::new()).bindings()
    }

    /// Vertices format
    fn vertices(&self) -> &'static [(&'static [Element<Format>], ElemStride)] {
        P::VERTICES
    }

    /// Load shaders
    fn shaders<'a>(
        &self,
        shaders: &'a mut SmallVec<[B::ShaderModule; 5]>,
        device: &B::Device,
    ) -> Result<GraphicsShaderSet<'a, B>, ShaderError> {
        P::shaders(shaders, device)
    }

    fn prepare(
        &mut self,
        pool: &mut DescriptorPool<B>,
        cbuf: &mut CommandBuffer<B, Transfer>,
        device: &B::Device,
        aux: &mut T,
    ) {
        P::prepare(
            self,
            pool,
            cbuf,
            device,
            aux,
        );
    }

    fn draw_inline<'a>(
        &mut self,
        layout: &B::PipelineLayout,
        encoder: RenderPassInlineEncoder<B>,
        device: &B::Device,
        aux: &T,
    ) {
        let binder = Binder::<B, P::Bindings>::new(layout, P::layout(Layout::new()));
        P::draw_inline(
            self,
            binder,
            encoder,
            device,
            aux,
        );
    }

    fn cleanup(&mut self, pool: &mut DescriptorPool<B>, device: &B::Device, aux: &mut T) {
        P::cleanup(self, pool, device, aux);
    }
}


/// Single node in rendering graph.
/// Nodes can use output of other nodes as input.
/// Such connection called `dependency`
#[derive(Debug)]
pub struct PassNode<B: Backend, T> {
    clears: Vec<ClearValue>,
    descriptors: DescriptorPool<B>,
    pipeline_layout: B::PipelineLayout,
    graphics_pipeline: B::GraphicsPipeline,
    render_pass: B::RenderPass,
    framebuffer: SuperFramebuffer<B>,
    pass: Box<AnyPass<B, T>>,
    pub(crate) depends: Option<(usize, PipelineStage)>,
}

impl<B, T> PassNode<B, T>
where
    B: Backend,
{
    /// Prepares to record actual drawing command.
    /// This is called outside of renderpass. And has exclusive acces for `T`.
    /// `PassNode::prepare` function is called sequentially for all passes.
    /// Thus it is required to be fast.
    /// 
    /// `cbuf` - command buffer to record transfer commands.
    /// `device` - to do stuff.
    /// `aux` - auxiliary data for passes.
    pub fn prepare<C>(
        &mut self,
        cbuf: &mut CommandBuffer<B, C>,
        device: &B::Device,
        aux: &mut T,
    ) where
        C: Supports<Transfer>,
    {
        // Run custom preparation
        // * Write descriptor sets
        // * Store caches
        // * Bind pipeline layout with descriptors sets
        self.pass.prepare(&mut self.descriptors, cbuf.downgrade(), device, aux);
    }

    /// Binds pipeline and renderpass to the command buffer `cbuf`.
    /// Executes `Pass::prepare` and `Pass::draw_inline` of the inner `Pass`
    /// to record transfer and draw commands.
    ///
    /// `cbuf` - command buffer to record drawing commands.
    /// `rect` - area to draw in.
    /// `frame` - specifies which framebuffer and descriptor sets to use.
    /// `device` - to do stuff.
    /// `aux` - auxiliary data for passes.
    pub fn draw_inline<C>(
        &mut self,
        cbuf: &mut CommandBuffer<B, C>,
        rect: Rect,
        frame: SuperFrame<B>,
        device: &B::Device,
        aux: &T,
    ) where
        C: Supports<Graphics>,
    {
        // Bind pipeline
        cbuf.bind_graphics_pipeline(&self.graphics_pipeline);

        let encoder = {
            // Begin render pass with single inline subpass
            cbuf.begin_renderpass_inline(
                &self.render_pass,
                pick(&self.framebuffer, frame),
                rect,
                &self.clears,
            )
        };

        // Record custom drawing calls
        self.pass.draw_inline(
            &self.pipeline_layout,
            encoder,
            device,
            aux,
        );
    }

    pub fn dispose(mut self, device: &B::Device, aux: &mut T) {
        self.pass.cleanup(&mut self.descriptors, device, aux);
        match self.framebuffer {
            SuperFramebuffer::Owned(framebuffers) => for framebuffer in framebuffers {
                device.destroy_framebuffer(framebuffer);
            },
            _ => {}
        }
        device.destroy_renderpass(self.render_pass);
        device.destroy_graphics_pipeline(self.graphics_pipeline);
        device.destroy_pipeline_layout(self.pipeline_layout);
    }
}
