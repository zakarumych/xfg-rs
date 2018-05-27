use std::{
    borrow::Borrow, marker::PhantomData, ops::Range, sync::{atomic::AtomicUsize, Arc},
};

use chain::resource::{Buffer, Id, Image};
use either::{Either, Left, Right};
use hal::{buffer, format::Format, image, pso::PipelineStage, window::Backbuffer, Backend};

/// Id of the image.
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct ImageId(pub(crate) Id<Image>);

impl ImageId {
    pub fn new(id: u32) -> ImageId {
        ImageId(Id::new(id))
    }

    pub fn index(&self) -> usize {
        self.0.index() as usize
    }
}

/// Id of the buffer.
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BufferId(pub(crate) Id<Buffer>);

impl BufferId {
    pub fn new(id: u32) -> BufferId {
        BufferId(Id::new(id))
    }

    pub fn index(&self) -> usize {
        self.0.index() as usize
    }
}

/// Set of barriers for the node to execute.
pub struct Barriers<S> {
    /// Buffer barriers.
    pub acquire: Option<Range<(S, PipelineStage)>>,

    /// Image barriers.
    pub release: Option<Range<(S, PipelineStage)>>,
}

/// Buffer info.
pub struct BufferInfo<'a, U: 'a> {
    /// Id of the buffer.
    pub id: BufferId,

    /// Barriers required for the buffer.
    pub barriers: Barriers<buffer::State>,

    /// Resource.
    pub resource: &'a BufferResource<U>,
}

pub struct BufferResource<U> {
    /// Size of the buffer.
    pub size: u64,

    /// The buffer.
    pub buffer: U,
}

/// Image info for particular `Node`.
pub struct ImageInfo<'a, I: 'a> {
    /// Id of the image.
    pub id: ImageId,

    /// Barriers required for the image.
    pub barriers: Barriers<image::State>,

    /// Layout in which the image is for the node.
    pub layout: image::Layout,

    /// Resource.
    pub resource: &'a ImageResource<I>,
}

pub struct ImageResource<I> {
    /// Kind of the image.
    pub kind: image::Kind,

    /// Format of the image.
    pub format: Format,

    /// The image.
    pub image: I,
}
