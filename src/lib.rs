
extern crate gfx_hal as hal;
extern crate gfx_chain as chain;

mod node;
mod graph;

pub use node::{Node, NodeDesc};

use std::{borrow::Borrow, cmp::Ordering};
use hal::{Backend, queue::{Capability, QueueFamily, QueueFamilyId, QueueType}};


fn pick_queue_family<B, C, F>(families: F) -> QueueFamilyId
where
    B: Backend,
    C: Capability,
    F: IntoIterator,
    F::Item: Borrow<B::QueueFamily>,
{
    families
        .into_iter()
        .filter(|qf| C::supported_by(qf.borrow().queue_type()))
        .min_by(|left, right| match (left.borrow().queue_type(), right.borrow().queue_type()) {
            (_, QueueType::General) => Ordering::Less,
            (QueueType::General, _) => Ordering::Greater,
            (QueueType::Transfer, _) => Ordering::Less,
            _ => Ordering::Equal,
        })
        .unwrap()
        .borrow()
        .id()
}
