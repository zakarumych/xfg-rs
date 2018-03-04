//! Defines attachments for the rendering `Graph`.
//!

use std::ops::Range;

use gfx_hal::command::{ClearColor, ClearDepthStencil, ClearValue};
use gfx_hal::format::Format;
use gfx_hal::image::{ImageLayout, Usage as ImageUsage};
use gfx_hal::pass::{AttachmentLoadOp, AttachmentStoreOp};

use chain::{ImageChain, ImageChainId, ImageInit};

/// Attachment declaration.
#[derive(Clone, Copy, Debug)]
pub struct Attachment {
    pub(crate) format: Format,
    pub(crate) init: ImageInit,
}

impl From<ColorAttachment> for Attachment {
    fn from(color: ColorAttachment) -> Self {
        color.0
    }
}

impl From<DepthStencilAttachment> for Attachment {
    fn from(depth_stencil: DepthStencilAttachment) -> Self {
        depth_stencil.0
    }
}

/// Attachment declaration with color format.
#[derive(Clone, Copy, Debug)]
pub struct ColorAttachment(pub(crate) Attachment);

impl ColorAttachment {
    /// Declare new attachment with format specified.
    ///
    /// # Panics
    ///
    /// If format aspect is not color.
    pub fn new(format: Format) -> Self {
        assert!(format.is_color());
        ColorAttachment(Attachment {
            format,
            init: ImageInit::DontCare,
        })
    }

    /// Set clearing color for the attachment.
    /// First pass that use the attachment as output will clear it.
    pub fn with_clear(mut self, clear: ClearColor) -> Self {
        self.set_clear(clear);
        self
    }

    /// Set clearing color for the attachment.
    /// First pass that use the attachment as output will clear it.
    pub fn set_clear(&mut self, clear: ClearColor) {
        self.0.init = ImageInit::Clear(ClearValue::Color(clear));
    }

    /// Set attachment to be loaded instead of cleared.
    pub fn load(mut self) -> Self {
        self.set_load();
        self
    }

    /// Set attachment to be loaded instead of cleared.
    pub fn set_load(&mut self) {
        self.0.init = ImageInit::Load;
    }
}

/// Attachment declaration with depth-stencil format.
#[derive(Clone, Copy, Debug)]
pub struct DepthStencilAttachment(pub(crate) Attachment);

impl DepthStencilAttachment {
    /// Declare new attachment with format specified.
    ///
    /// # Panics
    ///
    /// If format aspect doesn't contain depth.
    pub fn new(format: Format) -> Self {
        assert!(format.is_depth());
        DepthStencilAttachment(Attachment {
            format,
            init: ImageInit::DontCare,
        })
    }

    /// Set clearing values for the attachment.
    /// First pass that use the attachment as output will clear it.
    pub fn with_clear(mut self, clear: ClearDepthStencil) -> Self {
        self.set_clear(clear);
        self
    }

    /// Set clearing values for the attachment.
    /// First pass that use the attachment as output will clear it.
    pub fn set_clear(&mut self, clear: ClearDepthStencil) {
        self.0.init = ImageInit::Clear(ClearValue::DepthStencil(clear));
    }

    /// Set attachment to be loaded instead of cleared.
    pub fn load(mut self) -> Self {
        self.set_load();
        self
    }

    /// Set attachment to be loaded instead of cleared.
    pub fn set_load(&mut self) {
        self.0.init = ImageInit::Load;
    }
}

#[derive(Debug)]
pub(crate) struct AttachmentImages {
    pub(crate) images: Range<usize>,
    pub(crate) views: Range<usize>,
}

#[derive(Debug)]
pub(crate) struct AttachmentDesc {
    pub(crate) format: Format,
    pub(crate) init: ImageInit,
    pub(crate) writes: Option<(usize, usize)>,
    pub(crate) reads: Option<(usize, usize)>,
}

impl AttachmentDesc {
    pub(crate) fn is_read(&self, index: usize) -> bool {
        self.writes
            .map_or(false, |(first, _)| first == index && !self.init.discard())
            || self.writes
                .map_or(false, |(first, last)| first < index && last >= index)
            || self.reads
                .map_or(false, |(first, last)| first <= index && last >= index)
    }
    pub(crate) fn is_write(&self, index: usize) -> bool {
        self.writes
            .map_or(false, |(first, last)| first <= index && last >= index)
    }
}

pub type AttachmentRef = ImageChainId;
