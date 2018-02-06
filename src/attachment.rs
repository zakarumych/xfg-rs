//! Defines attachments for the rendering `Graph`.
//!

use std::ops::Range;
use std::ptr::eq;

use gfx_hal::command::{ClearColor, ClearDepthStencil};
use gfx_hal::format::{AspectFlags, Format};

/// Attachment declaration with color format.
#[derive(Debug)]
pub struct ColorAttachment {
    pub(crate) format: Format,
    pub(crate) clear: Option<ClearColor>,
}

impl ColorAttachment {
    /// Declare new attachment with format specified.
    ///
    /// # Panics
    ///
    /// If format aspect is not color.
    pub fn new(format: Format) -> Self {
        assert_eq!(format.aspect_flags(), AspectFlags::COLOR);
        ColorAttachment {
            format,
            clear: None,
        }
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
        self.clear = Some(clear);
    }

    pub(crate) fn ptr(&self) -> *const Self {
        self as *const _
    }
}

/// Attachment declaration with depth-stencil format.
#[derive(Debug)]
pub struct DepthStencilAttachment {
    pub(crate) format: Format,
    pub(crate) clear: Option<ClearDepthStencil>,
}

impl DepthStencilAttachment {
    /// Declare new attachment with format specified.
    ///
    /// # Panics
    ///
    /// If format aspect doesn't contain depth.
    pub fn new(format: Format) -> Self {
        assert!(format.aspect_flags().contains(AspectFlags::DEPTH));
        DepthStencilAttachment {
            format,
            clear: None,
        }
    }

    /// Set clearing values for the attachment.
    /// First pass that use the attachment as output will clear it.
    pub fn set_clear(&mut self, clear: ClearDepthStencil) {
        self.clear = Some(clear);
    }

    /// Set clearing values for the attachment.
    /// First pass that use the attachment as output will clear it.
    pub fn with_clear(mut self, clear: ClearDepthStencil) -> Self {
        self.set_clear(clear);
        self
    }

    pub(crate) fn ptr(&self) -> *const Self {
        self as *const _
    }
}

/// Reference to either color or depth-stencil attachment declaration.
#[derive(Clone, Copy, Debug)]
pub enum Attachment<'a> {
    /// Color attachment
    Color(&'a ColorAttachment),

    /// Depth-stencil attachment
    DepthStencil(&'a DepthStencilAttachment),
}

impl<'a> Attachment<'a> {
    /// Get format of the attachment.
    pub(crate) fn format(self) -> Format {
        match self {
            Attachment::Color(color) => color.format,
            Attachment::DepthStencil(depth) => depth.format,
        }
    }

    pub(crate) fn is(self, rhs: Self) -> bool {
        match (self, rhs) {
            (Attachment::Color(lhs), Attachment::Color(rhs)) => eq(lhs, rhs),
            (Attachment::DepthStencil(lhs), Attachment::DepthStencil(rhs)) => eq(lhs, rhs),
            _ => false,
        }
    }
}

impl<'a> From<&'a ColorAttachment> for Attachment<'a> {
    fn from(color: &'a ColorAttachment) -> Self {
        Attachment::Color(color)
    }
}

impl<'a> From<&'a DepthStencilAttachment> for Attachment<'a> {
    fn from(depth_stencil: &'a DepthStencilAttachment) -> Self {
        Attachment::DepthStencil(depth_stencil)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InputAttachmentDesc {
    pub(crate) format: Format,
    pub(crate) indices: Range<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct ColorAttachmentDesc {
    pub(crate) format: Format,
    pub(crate) indices: Option<Range<usize>>,
    pub(crate) clear: Option<ClearColor>,
}

#[derive(Clone, Debug)]
pub(crate) struct DepthStencilAttachmentDesc {
    pub(crate) format: Format,
    pub(crate) indices: Option<Range<usize>>,
    pub(crate) clear: Option<ClearDepthStencil>,
}
