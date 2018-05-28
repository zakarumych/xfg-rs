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

#[cfg(feature = "profile")]
extern crate flame;

#[cfg(feature = "profile")]
macro_rules! profile {
    ($name:tt) => {
        let guard = ::flame::start_guard(concat!("'", $name, "' at : ", line!()));
    }
}

#[cfg(not(feature = "profile"))]
macro_rules! profile {
    ($name:tt) => {}
}

mod graph;
mod node;
mod util;

pub use graph::{Graph, GraphBuilder};
pub use node::{build::NodeBuilder, present, render, Node, NodeDesc};
pub use util::{Barriers, BufferId, BufferInfo, BufferResource, ImageId, ImageInfo, ImageResource};

