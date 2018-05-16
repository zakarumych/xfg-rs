use chain::resource::{Buffer, Id, Image};

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct ImageId(pub(crate) Id<Image>);

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct BufferId(pub(crate) Id<Buffer>);
