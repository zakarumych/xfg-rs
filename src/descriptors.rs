use gfx_hal::{Backend, Device};
use gfx_hal::pso::{DescriptorPool as RawDescriptorPool, DescriptorRangeDesc,
                   DescriptorSetLayoutBinding, DescriptorType};

const CAPACITY: usize = 1024 * 32;

/// Simple growing wrapper for `Backend::DescriptorPool`.
///
/// ### Type parameters:
///
/// - `B`: hal `Backend`
#[derive(Debug)]
pub struct DescriptorPool<B: Backend> {
    range: Vec<DescriptorRangeDesc>,
    layout: B::DescriptorSetLayout,
    pools: Vec<B::DescriptorPool>,
    sets: Vec<B::DescriptorSet>,
    count: usize,
}

impl<B> DescriptorPool<B>
where
    B: Backend,
{
    /// Create a new descriptor pool for the given bindings list
    ///
    /// ### Parameters:
    ///
    /// - `bindings`: bindings given by a single `Pass`
    /// - `device`: graphics device
    pub fn new(bindings: &[DescriptorSetLayoutBinding], device: &B::Device) -> Self {
        let range = bindings_to_range_desc(bindings);
        DescriptorPool {
            layout: device.create_descriptor_set_layout(bindings),
            pools: Vec::new(),
            sets: Vec::new(),
            range,
            count: 0,
        }
    }

    /// Dispose of all allocated sets and pools.
    ///
    /// ### Parameters:
    ///
    /// - `device`: graphics device
    pub fn dispose(self, device: &B::Device) {
        assert_eq!(self.count, self.sets.len());
        #[cfg(feature = "gfx-metal")]
        {
            if device.downcast_ref::<::metal::Device>().is_none() {
                for pool in self.pools {
                    pool.reset();
                }
            }
        }
        drop(self.sets);
        device.destroy_descriptor_set_layout(self.layout);
    }

    /// Get descriptor set layout.
    pub fn layout(&self) -> &B::DescriptorSetLayout {
        &self.layout
    }

    /// Allocate a descriptor set on the given device.
    ///
    /// ### Parameters:
    ///
    /// - `device`: graphics device
    pub fn allocate(&mut self, device: &B::Device) -> B::DescriptorSet {
        if self.sets.is_empty() {
            // Check if there are sets available
            if self.count == self.pools.len() * CAPACITY {
                // Allocate new pool
                self.pools
                    .push(device.create_descriptor_pool(CAPACITY, &self.range));
            }
            self.count += 1;
            // allocate set
            self.pools
                .last_mut()
                .unwrap()
                .allocate_sets(&[&self.layout])
                .pop()
                .unwrap()
        } else {
            // get unused set
            self.sets.pop().unwrap()
        }
    }

    /// Free descriptor set
    ///
    /// ### Parameters:
    ///
    /// - `set`: descriptor set to free (returns the set to the pool)
    pub fn free(&mut self, set: B::DescriptorSet) {
        self.sets.push(set);
    }
}

fn bindings_to_range_desc(bindings: &[DescriptorSetLayoutBinding]) -> Vec<DescriptorRangeDesc> {
    let mut desc: Vec<DescriptorRangeDesc> = Vec::new();
    for binding in bindings {
        let desc_len = desc.len();
        desc.extend(
            (desc_len..binding.binding + 1).map(|_| DescriptorRangeDesc {
                ty: DescriptorType::UniformBuffer,
                count: 0,
            }),
        );
        desc[binding.binding].ty = binding.ty;
        desc[binding.binding].count = binding.count;
    }
    desc
}
