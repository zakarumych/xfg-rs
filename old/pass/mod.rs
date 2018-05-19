//! Defines the `Pass` trait, the main building block of the rendering `Graph`s

pub use self::build::PassBuilder;

use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

use gfx_hal::{Backend, Device, Primitive};
use gfx_hal::command::{ClearValue, CommandBuffer, Primary, RenderPassInlineEncoder};
use gfx_hal::device::ShaderError;
use gfx_hal::format::Format;
use gfx_hal::pso::{DescriptorSetLayoutBinding, ElemStride, Element, GraphicsShaderSet,
                   PipelineStage, Rasterizer, Viewport};
use gfx_hal::queue::capability::{Graphics, Supports, Transfer};

use smallvec::SmallVec;

use descriptors::DescriptorPool;
use frame::{pick, SuperFrame, SuperFramebuffer};

mod build;

/// Trait provide description of `Pass`.
pub trait PassDesc: Debug {
    /// Name of the pass
    fn name<'a>(&'a self) -> &'a str;

    /// Sampled images count.
    /// Pass will be able to sample data from those images
    /// in vertes and/or fragment shader.
    fn sampled(&self) -> usize;

    /// Storage images count.
    /// Pass will be able to load data from those images
    /// in vertes and/or fragment shader.
    fn storage(&self) -> usize;

    /// Input attachments count.
    /// Pass will be able to load data from those images
    /// in fragment shader but at the fragment’s (x, y, layer) framebuffer coordinates.
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

    /// Default primitve for the pass.
    fn primitive(&self) -> Primitive {
        Primitive::TriangleList
    }

    /// Default rasterizer for the pass.
    fn rasterizer(&self) -> Rasterizer {
        Rasterizer::FILL
    }

    /// Create builder
    fn build(self, viewport: Viewport) -> PassBuilder<Self>
    where
        Self: Sized,
    {
        PassBuilder::new(self, viewport)
    }
}

impl<P, Y> PassDesc for Y
where
    Y: Debug + Deref<Target = P>,
    P: PassDesc + ?Sized + 'static,
{
    fn name<'a>(&'a self) -> &str {
        P::name(self)
    }
    fn sampled(&self) -> usize {
        P::sampled(self)
    }
    fn storage(&self) -> usize {
        P::storage(self)
    }
    fn inputs(&self) -> usize {
        P::inputs(self)
    }
    fn colors(&self) -> usize {
        P::colors(self)
    }
    fn depth(&self) -> bool {
        P::depth(self)
    }
    fn stencil(&self) -> bool {
        P::stencil(self)
    }
    fn vertices(&self) -> &[(&[Element<Format>], ElemStride)] {
        P::vertices(self)
    }
    fn bindings(&self) -> &[DescriptorSetLayoutBinding] {
        P::bindings(self)
    }

    fn primitive(&self) -> Primitive {
        P::primitive(self)
    }

    fn rasterizer(&self) -> Rasterizer {
        P::rasterizer(self)
    }
}

/// Trait to load shaders for `Pass`.
pub trait PassShaders<B, D>: PassDesc
where
    B: Backend,
    D: BorrowMut<B::Device>,
{
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
        device: &mut D,
    ) -> Result<GraphicsShaderSet<'a, B>, ShaderError>;
}

impl<B, D, P, Y> PassShaders<B, D> for Y
where
    B: Backend,
    D: BorrowMut<B::Device>,
    Y: Debug + Deref<Target = P>,
    P: PassShaders<B, D> + ?Sized + 'static,
{
    fn shaders<'a>(
        &'a self,
        shaders: &'a mut SmallVec<[B::ShaderModule; 5]>,
        device: &mut D,
    ) -> Result<GraphicsShaderSet<'a, B>, ShaderError> {
        P::shaders(self, shaders, device)
    }
}

/// `Pass`es are the building blocks a rendering `Graph`.
///
/// `Pass` is similar in concept to `gfx_hal::Backend::RenderPass`.
/// But several `Pass`es may be mapped into a single `RenderPass` as subpasses.
/// A `Pass` defines its inputs and outputs, and also the required vertex format and bindings.
///
/// ### Type parameters:
///
/// - `B`: render `Backend`
/// - `D`: device
/// - `T`: auxiliary data used by the `Pass`, can be anything the `Pass` requires, such as meshes,
///        caches, etc
pub trait Pass<B, D, T>: PassShaders<B, D>
where
    B: Backend,
    D: BorrowMut<B::Device>,
{
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
    fn prepare(
        &mut self,
        pool: &mut DescriptorPool<B>,
        cbuf: &mut CommandBuffer<B, Transfer>,
        device: &mut D,
        inputs: &[&B::Image],
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
    fn draw_inline(
        &mut self,
        layout: &B::PipelineLayout,
        encoder: RenderPassInlineEncoder<B, Primary>,
        device: &D,
        inputs: &[&B::Image],
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
    fn cleanup(&mut self, pool: &mut DescriptorPool<B>, device: &mut D, aux: &mut T);
}

impl<B, D, T, P, Y> Pass<B, D, T> for Y
where
    B: Backend,
    D: BorrowMut<B::Device>,
    P: Pass<B, D, T> + ?Sized + 'static,
    Y: Debug + DerefMut<Target = P>,
{
    fn prepare<'a>(
        &mut self,
        pool: &mut DescriptorPool<B>,
        cbuf: &mut CommandBuffer<B, Transfer>,
        device: &mut D,
        inputs: &[&B::Image],
        frame: usize,
        aux: &mut T,
    ) {
        P::prepare(self, pool, cbuf, device, inputs, frame, aux)
    }

    fn draw_inline<'a>(
        &mut self,
        layout: &B::PipelineLayout,
        encoder: RenderPassInlineEncoder<B, Primary>,
        device: &D,
        inputs: &[&B::Image],
        frame: usize,
        aux: &T,
    ) {
        P::draw_inline(self, layout, encoder, device, inputs, frame, aux)
    }

    fn cleanup(&mut self, pool: &mut DescriptorPool<B>, device: &mut D, aux: &mut T) {
        P::cleanup(self, pool, device, aux)
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
pub(crate) struct PassNode<B: Backend, P> {
    clears: Vec<ClearValue>,
    descriptors: DescriptorPool<B>,
    pipeline_layout: B::PipelineLayout,
    graphics_pipeline: B::GraphicsPipeline,
    renderpass: B::RenderPass,
    framebuffer: SuperFramebuffer<B>,
    pass: P,
    inputs: Vec<Vec<usize>>,
    viewport: Viewport,
    pub(crate) depends: Option<(usize, PipelineStage)>,
}

impl<B, P> PassNode<B, P>
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
    pub fn prepare<C, D, T, I>(
        &mut self,
        cbuf: &mut CommandBuffer<B, C>,
        device: &mut D,
        images: &[I],
        frame: SuperFrame<B>,
        aux: &mut T,
    ) where
        C: Supports<Transfer>,
        D: BorrowMut<B::Device>,
        P: Pass<B, D, T>,
        I: Borrow<B::Image>,
    {
        let inputs = self.inputs
            .get(frame.index())
            .map_or(SmallVec::new(), |inputs| {
                inputs
                    .iter()
                    .map(|&index| images[index].borrow())
                    .collect::<SmallVec<[_; 16]>>()
            });

        // Run custom preparation
        // * Write descriptor sets
        // * Store caches
        // * Bind pipeline layout with descriptors sets
        self.pass.prepare(
            &mut self.descriptors,
            cbuf.downgrade(),
            device,
            &inputs,
            frame.index(),
            aux,
        );
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
    pub fn draw_inline<C, D, T, I>(
        &mut self,
        cbuf: &mut CommandBuffer<B, C>,
        device: &D,
        images: &[I],
        frame: SuperFrame<B>,
        aux: &T,
    ) where
        C: Supports<Graphics>,
        D: BorrowMut<B::Device>,
        P: Pass<B, D, T>,
        I: Borrow<B::Image>,
    {
        cbuf.set_viewports(0, &[self.viewport.clone()]);
        cbuf.set_scissors(0, &[self.viewport.rect]);

        // Bind pipeline
        cbuf.bind_graphics_pipeline(&self.graphics_pipeline);

        let encoder = {
            // Begin render pass with single inline subpass
            cbuf.begin_render_pass_inline(
                &self.renderpass,
                pick(&self.framebuffer, &frame),
                self.viewport.rect,
                &self.clears,
            )
        };

        let inputs = self.inputs
            .get(frame.index())
            .map_or(SmallVec::new(), |inputs| {
                inputs
                    .iter()
                    .map(|&index| images[index].borrow())
                    .collect::<SmallVec<[_; 16]>>()
            });

        // Record custom drawing calls
        self.pass.draw_inline(
            &self.pipeline_layout,
            encoder,
            device,
            &inputs,
            frame.index(),
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
    pub fn dispose<D, T>(mut self, device: &mut D, aux: &mut T)
    where
        D: BorrowMut<B::Device>,
        P: Pass<B, D, T>,
    {
        self.pass.cleanup(&mut self.descriptors, device, aux);
        match self.framebuffer {
            SuperFramebuffer::Owned(framebuffers) => for framebuffer in framebuffers {
                device.borrow_mut().destroy_framebuffer(framebuffer);
            },
            _ => {}
        }
        device
            .borrow_mut()
            .destroy_graphics_pipeline(self.graphics_pipeline);
        // ::std::mem::forget(self.graphics_pipeline);
        device.borrow_mut().destroy_render_pass(self.renderpass);
        device
            .borrow_mut()
            .destroy_pipeline_layout(self.pipeline_layout);
    }
}