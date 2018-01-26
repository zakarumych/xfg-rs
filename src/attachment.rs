//! Defines attachments for the rendering `Graph`.
//!

use std::ptr::eq;

use gfx_hal::Backend;
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

#[derive(Debug)]
pub(crate) enum AttachmentImageViews<'a, B: Backend> {
    Owned(&'a [B::ImageView]),
    External,
}

#[derive(Debug)]
pub(crate) struct InputAttachmentDesc<'a, B: Backend> {
    pub(crate) format: Format,
    pub(crate) view: &'a [B::ImageView],
}

#[derive(Debug)]
pub(crate) struct ColorAttachmentDesc<'a, B: Backend> {
    pub(crate) format: Format,
    pub(crate) view: AttachmentImageViews<'a, B>,
    pub(crate) clear: Option<ClearColor>,
}

#[derive(Debug)]
pub(crate) struct DepthStencilAttachmentDesc<'a, B: Backend> {
    pub(crate) format: Format,
    pub(crate) view: AttachmentImageViews<'a, B>,
    pub(crate) clear: Option<ClearDepthStencil>,
}
