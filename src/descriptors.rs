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
        drop(self.sets);
        // TODO: Uncomment when all backends will support this.
        // for pool in self.pools {
        //     pool.reset();
        // }
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
                .allocate_sets(Some(&self.layout))
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
    let cast = |i: usize| match i {
        x if x == DescriptorType::Sampler as usize => DescriptorType::Sampler,
        x if x == DescriptorType::CombinedImageSampler as usize => {
            DescriptorType::CombinedImageSampler
        }
        x if x == DescriptorType::SampledImage as usize => DescriptorType::SampledImage,
        x if x == DescriptorType::StorageImage as usize => DescriptorType::StorageImage,
        x if x == DescriptorType::UniformTexelBuffer as usize => DescriptorType::UniformTexelBuffer,
        x if x == DescriptorType::StorageTexelBuffer as usize => DescriptorType::StorageTexelBuffer,
        x if x == DescriptorType::UniformBuffer as usize => DescriptorType::UniformBuffer,
        x if x == DescriptorType::StorageBuffer as usize => DescriptorType::StorageBuffer,
        x if x == DescriptorType::UniformBufferDynamic as usize => {
            DescriptorType::UniformBufferDynamic
        }
        x if x == DescriptorType::UniformImageDynamic as usize => {
            DescriptorType::UniformImageDynamic
        }
        x if x == DescriptorType::InputAttachment as usize => DescriptorType::InputAttachment,
        _ => unreachable!(),
    };

    let mut descs: Vec<DescriptorRangeDesc> = vec![];
    for binding in bindings {
        let len = descs.len();
        descs.extend(
            (len..(binding.ty as usize) + 1).map(|ty| DescriptorRangeDesc {
                ty: cast(ty),
                count: 0,
            }),
        );
        let ref mut desc = descs[binding.ty as usize];
        debug_assert_eq!(desc.ty, binding.ty);
        desc.count += binding.count * CAPACITY;
    }
    descs.retain(|desc| desc.count > 0);
    descs
}
