use std::borrow::Borrow;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Range};

use gfx_hal::buffer::{Access as BufferAccess, Usage as BufferUsage, State as BufferState};
use gfx_hal::command::ClearValue;
use gfx_hal::image::{Access as ImageAccess, ImageLayout, Usage as ImageUsage, State as ImageState};
use gfx_hal::pass::{AttachmentLoadOp, AttachmentStoreOp};
use gfx_hal::pso::PipelineStage;
use gfx_hal::queue::QueueFamilyId;

/// Image clearing.
#[derive(Clone, Copy, Debug)]
pub enum ImageInit {
    Clear(ClearValue),
    Load,
    DontCare,
}

impl ImageInit {
    pub fn load_op(&self) -> AttachmentLoadOp {
        match *self {
            ImageInit::Clear(_) => AttachmentLoadOp::Clear,
            ImageInit::Load => AttachmentLoadOp::Load,
            ImageInit::DontCare => AttachmentLoadOp::DontCare,
        }
    }

    pub fn discard(&self) -> bool {
        match *self {
            ImageInit::Load => false,
            _ => true,
        }
    }

    pub fn clear_value(&self) -> Option<ClearValue> {
        match *self {
            ImageInit::Clear(value) => Some(value),
            _ => None,
        }
    }
}

/// Unique identifier of the queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct QueueId(pub usize, pub QueueFamilyId);

/// Unique identifier for resource dependency chain.
/// Multiple resource can be associated with single chain
/// if all passes uses them the same way.
#[derive(Derivative)]
#[derivative(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ChainId<S>(usize, PhantomData<S>);
impl<S> ChainId<S> {
    pub(crate) fn new(index: usize) -> Self {
        ChainId(index, PhantomData)
    }
    pub(crate) fn index(&self) -> usize {
        self.0
    }
}

trait State {
    fn compatible(lhs: Self, rhs: Self) -> bool;
}

pub type BufferChainId = ChainId<(BufferState, BufferUsage)>;
pub type ImageChainId = ChainId<(ImageState, ImageUsage)>;

/// Piece of `Chain` of the resource dependency chain associated with pass.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Link<S> {
    pub id: ChainId<S>,
    pub stages: PipelineStage,
    pub state: S,
}

/// Piece of `Chain` of the buffer dependency chain associated with pass.
pub type BufferLink = Link<(BufferState, BufferUsage)>;

/// Piece of `Chain` of the image dependency chain associated with pass.
pub type ImageLink = Link<(ImageState, ImageUsage)>;

/// All links pass defines.
#[derive(Clone, Debug)]
pub struct PassLinks<S> {
    pub queue: QueueId,
    pub links: Vec<Link<S>>,
}

#[derive(Clone, Debug)]
enum LinkTransition<S> {
    None,
    Barrier {
        states: Range<S>
    },
    Ownership {
        states: Range<S>,
        semaphore: usize,
    },
}

#[derive(Clone, Debug)]
struct ChainLink<S> {
    queue: QueueId,
    stages: PipelineStage,
    state: S,
    acquire: LinkTransition<S>,
    release: LinkTransition<S>,
}

/// Full dependency chain of the resource.
#[derive(Clone, Debug)]
pub struct Chain<S, U, I> {
    pub(crate) usage: U,
    pub(crate) init: I,
    pub(crate) links: Vec<Option<ChainLink<S>>>,
}

/// Full dependency chain of the buffer.
pub type BufferChain = Chain<BufferState, BufferUsage, ()>;

/// Full dependency chain of the image.
pub type ImageChain = Chain<ImageState, ImageUsage, ImageInit>;

impl<S, U, I> Chain<S, U, I> {
    fn new<P>(id: ChainId<S>, mut usage: U, iter: P) -> Self
    where
        P: IntoIterator,
        P::Item: Borrow<PassLinks<S>>,
    {
        let mut links = Vec::new();

        // Walk over passes
        for pass in iter {
            // Get links of the pass
            if let Some(link) = pass.borrow().links.iter().find(|l| l.id == id) {
                
                // Find last access
                for prev in links.iter().rev().filter_map(|l| l.map(|l| l.release )) {
                    
                }

                let link = ChainLink {
                    queue: pass.borrow().queue,
                    stages: l.stages,
                    state: l.state.0,
                    acquire: LinkTransition::None,
                    release: LinkTransition::None,
                };
                links.push(Some(link));
                usage |= l.state.1;
            }
        }
        Chain {
            links,
            usage,
            init: (),
        }
    }
}

impl ImageChain {
    pub fn load_op(&self, index: usize) -> AttachmentLoadOp {
        for p in &self.links[..index] {
            if p.is_some() {
                return AttachmentLoadOp::Load;
            }
        }
        self.init.load_op()
    }

    pub fn store_op(&self, index: usize) -> AttachmentStoreOp {
        for p in &self.links[index + 1..] {
            if p.is_some() {
                return AttachmentStoreOp::Store;
            }
        }
        AttachmentStoreOp::DontCare
    }

    pub fn pass_layout_transition(&self, index: usize) -> Range<ImageLayout> {
        let start = if self.links[..index]
            .iter()
            .filter_map(Option::as_ref)
            .count() == 0 && self.init.discard()
        {
            ImageLayout::Undefined
        } else {
            self.subpass_layout(index)
        };

        let end = if let Some(next) = self.links[index + 1..]
            .iter()
            .filter_map(Option::as_ref)
            .next()
        {
            next.state.1
        } else {
            self.subpass_layout(index)
        };

        start..end
    }

    pub fn subpass_layout(&self, index: usize) -> ImageLayout {
        (self.links[index].unwrap().state).1
    }

    pub fn clear_value(&self, index: usize) -> Option<ClearValue> {
        match self.load_op(index) {
            AttachmentLoadOp::Clear => self.init.clear_value(),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GraphChains {
    buffers: Vec<BufferChain>,
    images: Vec<ImageChain>,
}

impl GraphChains {
    pub(crate) fn new(buffers: usize, images: usize, links: &[PassLinks]) -> GraphChains {
        GraphChains {
            buffers: (0..buffers)
                .map(|i| BufferChain::new(ChainId::new(i), links))
                .collect(),
            images: (0..images)
                .map(|i| ImageChain::new(ChainId::new(i), links))
                .collect(),
        }
    }

    pub fn buffer(&self, id: BufferChainId) -> &BufferChain {
        &self.buffers[id.0]
    }

    pub fn image(&self, id: ImageChainId) -> &ImageChain {
        &self.images[id.0]
    }

    pub fn buffer_mut(&mut self, id: BufferChainId) -> &mut BufferChain {
        &mut self.buffers[id.0]
    }

    pub fn image_mut(&mut self, id: ImageChainId) -> &mut ImageChain {
        &mut self.images[id.0]
    }
}

impl Index<BufferChainId> for GraphChains {
    type Output = BufferChain;
    fn index(&self, index: BufferChainId) -> &BufferChain {
        self.buffer(index)
    }
}

impl Index<ImageChainId> for GraphChains {
    type Output = ImageChain;
    fn index(&self, index: ImageChainId) -> &ImageChain {
        self.image(index)
    }
}

impl IndexMut<BufferChainId> for GraphChains {
    fn index_mut(&mut self, index: BufferChainId) -> &mut BufferChain {
        self.buffer_mut(index)
    }
}

impl IndexMut<ImageChainId> for GraphChains {
    fn index_mut(&mut self, index: ImageChainId) -> &mut ImageChain {
        self.image_mut(index)
    }
}
