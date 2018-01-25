#[macro_use]
extern crate derivative;
extern crate gfx_hal;
extern crate relevant;
extern crate smallvec;

pub use attachment::{Attachment, ColorAttachment, DepthStencilAttachment};
pub use pass::{Pass, PassBuilder};
pub use graph::{Graph, GraphBuilder, GraphBuildError};
pub use descriptors::DescriptorPool;
pub use bindings::{Binding, BindingsList, Binder, SetBinder};

mod attachment;
mod bindings;
mod descriptors;
mod graph;
mod pass;
mod frame;
