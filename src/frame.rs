
use gfx_hal::Backend;
use gfx_hal::window::{Backbuffer, Frame};

/// This wrapper allow to abstract over two cases.
/// `Index` => index to one of multiple `Framebuffer`s created by the engine.
/// `Buffer` => Single `Framebuffer` associated with `Swapchain`.
#[derive(Derivative)]
#[derivative(Clone, Debug)]
pub enum SuperFrame<'a, B: Backend> {
    Index(usize),
    Buffer(&'a B::Framebuffer),
}

impl<'a, B> SuperFrame<'a, B>
where
    B: Backend,
{
    /// Create `SuperFrame` from `Backbuffer` and `Frame` index.
    pub fn new(backbuffer: &'a Backbuffer<B>, frame: Frame) -> Self {
        // Check if we have `Framebuffer` from `Surface` (usually with OpenGL backend) or `Image`s
        // In case it's `Image`s we need to pick `Framebuffer` for `RenderPass`es
        // that renders onto surface.
        match *backbuffer {
            Backbuffer::Images(_) => SuperFrame::Index(frame.id()),
            Backbuffer::Framebuffer(ref single) => SuperFrame::Buffer(single),
        }
    }
}

/// Framebuffer wrapper
#[derive(Debug)]
pub enum SuperFramebuffer<B: Backend> {
    /// Target is multiple `Framebuffer`s created over `ImageView`s
    Owned(Vec<B::Framebuffer>),

    /// Target is single `Framebuffer` associated with `Swapchain`
    External,
}

/// Picks correct framebuffer
pub fn pick<'a, B>(framebuffer: &'a SuperFramebuffer<B>, frame: SuperFrame<'a, B>) -> &'a B::Framebuffer
where
    B: Backend,
{
    use self::SuperFrame::*;
    use self::SuperFramebuffer::*;
    match (framebuffer, frame) {
        (&Owned(ref framebuffers), Index(index)) => &framebuffers[index],
        (&Owned(ref framebuffers), Buffer(_)) => &framebuffers[0],
        (&External, Buffer(ref framebuffer)) => framebuffer,
        _ => unreachable!("This combination can't happen"),
    }
}
