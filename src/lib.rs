#[macro_use]
extern crate derivative;
extern crate gfx_hal;
extern crate relevant;
extern crate smallvec;

pub use attachment::{Attachment, ColorAttachment, DepthStencilAttachment};
pub use bindings::{Binder, Binding, BindingsList, SetBinder};
pub use descriptors::DescriptorPool;
pub use graph::{Graph, GraphBuildError, GraphBuilder};
pub use pass::{Pass, PassBuilder};

mod attachment;
mod bindings;
mod descriptors;
mod graph;
mod pass;
mod frame;
