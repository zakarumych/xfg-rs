//! eXtensible FrameGraph for `gfx_hal`
//!
//! Provides instruments for building complex framegraphs that can orchestrate
//! command recording and synchronize access to attachments.
//!
//! User is still responsible for synchronizing access to other resources.
//!

// #![deny(dead_code)]
// #![deny(missing_docs)]
// #![deny(unused_imports)]
#![deny(unused_must_use)]

#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate derivative;
extern crate gfx_hal;
#[macro_use]
extern crate log;
extern crate relevant;
extern crate smallvec;

pub use attachment::{Attachment, ColorAttachment, DepthStencilAttachment};
pub use descriptors::DescriptorPool;
pub use frame::SuperFrame;
pub use graph::{Graph, GraphBuildError, GraphBuilder};
pub use pass::{Pass, PassBuilder, PassDesc, PassShaders};

mod attachment;
mod chain;
mod descriptors;
mod graph;
mod pass;
mod frame;
mod utils;
