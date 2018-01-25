//! Defines the `Pass` trait, the main building block of the rendering `Graph`s

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

mod build;

/// `Pass`es are the building blocks a rendering `Graph`.
///
/// `Pass` is similar in concept to `gfx_hal::Backend::RenderPass`.
/// But several `Pass`es may be mapped into a single `RenderPass` as subpasses.
/// A `Pass` defines its inputs and outputs, and also the required vertex format and bindings.
///
/// ### Type parameters:
///
/// - `B`: render `Backend`
/// - `T`: auxiliary data used by the `Pass`, can be anything the `Pass` requires, such as meshes,
///        caches, etc
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

    /// Will the pass write to the depth buffer
    const DEPTH: bool;

    /// Will the pass use the stencil buffer
    const STENCIL: bool;

    /// Vertex formats required by the pass
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
    /// This function gets called during the `Graph` build process, and is expected to load the
    /// shaders used by the pass.
    ///
    /// ### Parameters
    ///
    /// - `shaders`: any `ShaderModule` created by the pass should be added here, this is required
    ///              so the rendering `Graph` can destroy the shader modules after they have been
    ///              uploaded to the graphics device
    /// - `device`: graphics device
    ///
    /// ### Returns
    ///
    /// A set of `EntryPoint`s for the pass shaders.
    fn shaders<'a>(
        shaders: &'a mut SmallVec<[B::ShaderModule; 5]>,
        device: &B::Device,
    ) -> Result<GraphicsShaderSet<'a, B>, ShaderError>;

    /// Make preparation for actual drawing commands.
    ///
    /// Examples of tasks that should be performed during the preparation phase:
    ///  - Bind buffers
    ///  - Transfer data to graphics memory
    /// Note that pass has exclusive access to `T` during the preparation phase, which is executed
    /// sequentially and expected to be fast.
    ///
    /// ### Parameters:
    ///
    /// - `pool`: descriptor pool to use
    /// - `cbuf`: command buffer to record commands to
    /// - `device`: graphics device
    /// - `aux`: auxiliary data
    ///
    /// ### Type parameters:
    ///
    /// - `C`: Hal `Capability`
    fn prepare<'a, C>(
        &mut self,
        pool: &mut DescriptorPool<B>,
        cbuf: &mut CommandBuffer<B, C>,
        device: &B::Device,
        aux: &mut T,
    ) where
        C: Supports<Transfer>;

    /// Record actual drawing commands in inline fashion.
    ///
    /// Drawing methods define how to pick data for drawing and record drawing commands.
    /// During the drawing phase `T` is shared as passes record drawing commands in parallel.
    ///
    /// ### Parameters:
    ///
    /// - `binder`: binder used to bind bindings to descriptor sets
    /// - `encoder`: encoder used to record drawing commands
    /// - `device`: graphics device
    /// - `aux`: auxiliary data
    fn draw_inline<'a>(
        &mut self,
        binder: Binder<B, Self::Bindings>,
        encoder: RenderPassInlineEncoder<B>,
        device: &B::Device,
        aux: &T,
    );

    /// Cleanup before dropping this pass
    ///
    /// ### Parameters:
    ///
    /// - `pool`: descriptor pool used for this pass in the rendering graph
    /// - `device`: graphics device
    /// - `aux`: Auxiliary pass data, if the pass have anything stored there that needs to be
    ///          disposed
    fn cleanup(&mut self, pool: &mut DescriptorPool<B>, device: &B::Device, aux: &mut T);
}

/// Object-safe trait that mirrors the `Pass` trait.
/// Is blanket implemented for any type that implements `Pass`.
pub(crate) trait AnyPass<B, T>: Debug
where
    B: Backend,
{
    /// Name of the pass
    fn name(&self) -> &'static str;

    /// Input attachments desired format.
    /// Format of actual attachment may be different.
    /// It may be larger if another consumer expects larger format.
    /// It may be smaller because of hardware limitations.
    fn inputs(&self) -> usize;

    /// Number of colors to write
    fn colors(&self) -> usize;

    /// Will the pass write to the depth buffer
    fn depth(&self) -> bool;

    /// Will the pass use the stencil buffer
    fn stencil(&self) -> bool;

    /// Bindings for the descriptor sets used by the pass
    fn bindings(&self) -> SmallVec<[DescriptorSetLayoutBinding; 64]>;

    /// Vertex formats required by the pass
    fn vertices(&self) -> &'static [(&'static [Element<Format>], ElemStride)];

    /// Reflects [`Pass::shaders`] function
    ///
    /// [`Pass::shaders`]: trait.Pass.html#tymethod.shaders
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

    /// Reflects [`Pass::cleanup`] function
    ///
    /// [`Pass::cleanup`]: trait.Pass.html#tymethod.cleanup
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


/// Single node in the rendering graph.
/// Nodes can use output of other nodes as input, such a connection is called a `dependency`.
///
/// ### Type parameters:
///
/// - `B`: render `Backend`
/// - `T`: auxiliary data used by the `Pass`, can be anything the `Pass` requires, such as meshes,
///        caches, etc
#[derive(Debug)]
pub(crate) struct PassNode<B: Backend, T> {
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
    /// Prepares to record actual drawing commands.
    /// This is called outside of renderpass, and has exclusive access to `T`.
    /// `PassNode::prepare` function is called sequentially for all passes, and is expected to be
    /// fast.
    ///
    /// ### Parameters:
    ///
    /// - `cbuf`: command buffer to record transfer commands to
    /// - `device`: graphics device
    /// - `aux`: auxiliary data for the inner `Pass`
    ///
    /// ### Type parameters:
    ///
    /// - `C`: hal `Capability`
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
    /// Executes `Pass::draw_inline` of the inner `Pass` to record commands.
    ///
    /// ### Parameters:
    ///
    /// - `cbuf`: command buffer to record commands to
    /// - `rect`: area to draw in
    /// - `frame`: specifies which framebuffer and descriptor sets to use
    /// - `device`: graphics device
    /// - `aux`: auxiliary data for the inner `Pass`
    ///
    /// ### Type parameters:
    ///
    /// - `C`: hal `Capability`
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

    /// Dispose of all internal data created by the pass.
    ///
    /// Will call [`Pass::cleanup`], and destroy any locally created framebuffers, renderpasses,
    /// layouts, pipelines etc
    ///
    /// ### Parameters:
    ///
    /// - `device`: graphics device
    /// - `aux`: auxiliary data for the inner `Pass`
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
