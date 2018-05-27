#![allow(dead_code)]
#![allow(unused_imports)]
#![deny(unused_must_use)]

extern crate either;
extern crate gfx_chain as chain;
extern crate gfx_hal as hal;
#[macro_use]
extern crate log;
extern crate relevant;
extern crate smallvec;

mod graph;
mod node;
mod util;

pub use graph::{Graph, GraphBuilder};
pub use util::{Barriers, BufferInfo, ImageInfo, BufferResource, ImageResource, BufferId, ImageId};
pub use node::{Node, NodeDesc, build::NodeBuilder, render, present};
