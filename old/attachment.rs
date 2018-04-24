//! Defines attachments for the rendering `Graph`.
//!

use std::ops::Range;

use gfx_hal::command::{ClearColor, ClearDepthStencil, ClearValue};
use gfx_hal::format::Format;
use gfx_hal::image::{Layout, Usage as ImageUsage};
use gfx_hal::pass::{AttachmentLoadOp, AttachmentStoreOp};

/// Attachment declaration.
#[derive(Clone, Copy, Debug)]
pub struct Attachment {
    pub(crate) format: Format,
    pub(crate) clear: Option<ClearValue>,
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
            clear: None,
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
        self.0.clear = Some(ClearValue::Color(clear));
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
            clear: None,
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
        self.0.clear = Some(ClearValue::DepthStencil(clear));
    }
}

/// Reference to either color or depth-stencil attachment declaration in `GraphBuilder`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttachmentRef(pub(crate) usize);
impl AttachmentRef {
    pub(crate) fn index(&self) -> usize {
        self.0
    }
}

#[derive(Debug)]
pub(crate) struct AttachmentDesc {
    pub(crate) format: Format,
    pub(crate) clear: Option<ClearValue>,
    pub(crate) write: Option<Range<usize>>,
    pub(crate) read: Option<Range<usize>>,
    pub(crate) images: Option<Range<usize>>,
    pub(crate) views: Option<Range<usize>>,
    pub(crate) is_surface: bool,
    pub(crate) usage: ImageUsage,
}

impl AttachmentDesc {
    fn is_first_write(&self, index: usize) -> bool {
        self.write.clone().map_or(false, |w| w.start == index)
    }
    fn is_last_write(&self, index: usize) -> bool {
        self.write.clone().map_or(false, |w| w.end == index)
    }
    fn is_last_read(&self, index: usize) -> bool {
        self.read.clone().map_or(false, |r| r.end == index)
    }
    fn is_first_touch(&self, index: usize) -> bool {
        self.is_first_write(index)
    }
    fn is_last_touch(&self, index: usize) -> bool {
        self.is_last_read(index) || (self.is_last_write(index) && self.read.is_none())
    }

    pub(crate) fn load_op(&self, index: usize) -> AttachmentLoadOp {
        if self.is_first_touch(index) {
            if self.clear.is_some() {
                AttachmentLoadOp::Clear
            } else {
                AttachmentLoadOp::DontCare
            }
        } else {
            AttachmentLoadOp::Load
        }
    }

    pub(crate) fn store_op(&self, index: usize) -> AttachmentStoreOp {
        if self.is_last_touch(index) && !self.is_surface {
            if self.is_last_write(index) && !self.format.is_depth() {
                warn!(
                    "Pass at index {} writes to an attachment and nobody reads it",
                    index
                );
            }
            AttachmentStoreOp::DontCare
        } else {
            AttachmentStoreOp::Store
        }
    }

    pub(crate) fn image_layout_transition(&self, index: usize) -> Range<Layout> {
        let start = if self.is_first_touch(index) {
            Layout::Undefined
        } else {
            Layout::General
        };
        let end = if self.is_last_touch(index) && self.is_surface {
            Layout::Present
        } else {
            Layout::General
        };
        start..end
    }
}
