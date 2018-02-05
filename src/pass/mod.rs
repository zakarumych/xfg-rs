//! Defines the `Pass` trait, the main building block of the rendering `Graph`s

pub use self::build::PassBuilder;

use std::fmt::Debug;
use std::ops::Range;

use gfx_hal::{Backend, Device};
use gfx_hal::command::{ClearValue, CommandBuffer, Primary, Rect, RenderPassInlineEncoder};
use gfx_hal::device::ShaderError;
use gfx_hal::format::Format;
use gfx_hal::pso::{DescriptorSetLayoutBinding, ElemStride, Element, GraphicsShaderSet,
                   PipelineStage};
use gfx_hal::queue::capability::{Graphics, Supports, Transfer};

use smallvec::SmallVec;

use descriptors::DescriptorPool;
use frame::{pick, SuperFrame, SuperFramebuffer};

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
    fn name(&self) -> &str;

    /// Input attachments desired format.
    ///
    /// Format of the actual attachment may be different, it may be larger if another consumer
    /// expects a larger format, or smaller because of hardward limitations.
    fn inputs(&self) -> usize;

    /// Number of colors to write
    fn colors(&self) -> usize;

    /// Will the pass write to the depth buffer
    fn depth(&self) -> bool;

    /// Will the pass use the stencil buffer
    fn stencil(&self) -> bool;

    /// Vertex formats required by the pass
    fn vertices(&self) -> &[(&[Element<Format>], ElemStride)];

    /// Bindings for the descriptor sets used by the pass
    fn bindings(&self) -> &[DescriptorSetLayoutBinding];

    /// Build render pass
    fn build<'a>(self) -> PassBuilder<'static, B, T>
    where
        Self: Sized + 'static,
    {
        PassBuilder::new(self)
    }

    /// Load shaders
    ///
    /// This function gets called during the `Graph` build process, and is expected to load the
    /// shaders used by the pass.
    ///
    /// ### Parameters
    ///
    /// - `shaders`: `ShaderModule` objects created by the pass can be added here, if they are
    ///               not stored in the Pass. If they are added here, they will be destroyed by
    ///               the `Graph` after having been uploaded to the graphics device.
    /// - `device`: graphics device
    ///
    /// ### Returns
    ///
    /// A set of `EntryPoint`s for the pass shaders.
    fn shaders<'a>(
        &'a self,
        shaders: &'a mut SmallVec<[B::ShaderModule; 5]>,
        device: &B::Device,
    ) -> Result<GraphicsShaderSet<'a, B>, ShaderError>;

    /// Make preparation for actual drawing commands.
    ///
    /// Examples of tasks that should be performed during the preparation phase:
    ///  - Bind buffers
    ///  - Transfer data to graphics memory
    /// Note that the pass has exclusive access to `T` during the preparation phase, which is
    /// executed sequentially and expected to be fast.
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
    fn prepare<'a>(
        &mut self,
        pool: &mut DescriptorPool<B>,
        cbuf: &mut CommandBuffer<B, Transfer>,
        device: &B::Device,
        frame: usize,
        aux: &mut T,
    );

    /// Record actual drawing commands in inline fashion.
    ///
    /// Drawing methods define how to pick data for drawing and record drawing commands.
    /// During the drawing phase `T` is shared as passes record drawing commands in parallel.
    ///
    /// ### Parameters:
    ///
    /// - `layout`: pipeline layout
    /// - `encoder`: encoder used to record drawing commands
    /// - `device`: graphics device
    /// - `aux`: auxiliary data
    fn draw_inline<'a>(
        &mut self,
        layout: &B::PipelineLayout,
        encoder: RenderPassInlineEncoder<B, Primary>,
        device: &B::Device,
        inputs: &[&B::ImageView],
        frame: usize,
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
    renderpass: B::RenderPass,
    framebuffer: SuperFramebuffer<B>,
    pass: Box<Pass<B, T>>,
    pub(crate) depends: Option<(usize, PipelineStage)>,
    pub(crate) inputs: Vec<Range<usize>>,
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
    pub fn prepare<C>(&mut self, cbuf: &mut CommandBuffer<B, C>, device: &B::Device, frame: usize, aux: &mut T)
    where
        C: Supports<Transfer>,
    {
        // Run custom preparation
        // * Write descriptor sets
        // * Store caches
        // * Bind pipeline layout with descriptors sets
        self.pass
            .prepare(&mut self.descriptors, cbuf.downgrade(), device, frame, aux);
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
                &self.renderpass,
                pick(&self.framebuffer, &frame),
                rect,
                &self.clears,
            )
        };

        // Record custom drawing calls
        self.pass
            .draw_inline(&self.pipeline_layout, encoder, device, &[], frame.index(), aux);
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
        device.destroy_renderpass(self.renderpass);
        device.destroy_graphics_pipeline(self.graphics_pipeline);
        device.destroy_pipeline_layout(self.pipeline_layout);
    }
}
