use gfx_hal::buffer::Access as BufferAccess;
use gfx_hal::image::{Access as ImageAccess, ImageLayout};
use gfx_hal::pso::{DescriptorType, PipelineStage};

pub fn merge_image_layout<L>(layout: ImageLayout, layouts: L) -> ImageLayout
where
    L: IntoIterator<Item = ImageLayout>,
{
    layouts
        .into_iter()
        .fold(layout, |layout, next| common_image_layout(layout, next))
}

pub fn common_image_layout(left: ImageLayout, right: ImageLayout) -> ImageLayout {
    match (left, right) {
        (x, y) if x == y => x,
        (ImageLayout::Present, _) | (_, ImageLayout::Present) => {
            panic!("Present layout is unexpected here")
        }
        (ImageLayout::ShaderReadOnlyOptimal, ImageLayout::DepthStencilReadOnlyOptimal)
        | (ImageLayout::DepthStencilReadOnlyOptimal, ImageLayout::ShaderReadOnlyOptimal) => {
            ImageLayout::DepthStencilReadOnlyOptimal
        }
        (_, _) => ImageLayout::General,
    }
}

fn descriptor_type(i: usize) -> DescriptorType {
    match i {
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
        _ => panic!("No such descriptor type"),
    }
}


pub fn image_access_supported_pipeline_stages(access: ImageAccess) -> PipelineStage {
    type PS = PipelineStage;
    type A = ImageAccess;

    match access {
        A::COLOR_ATTACHMENT_READ | A::COLOR_ATTACHMENT_WRITE => PS::COLOR_ATTACHMENT_OUTPUT,
        A::TRANSFER_READ | A::TRANSFER_WRITE => PS::TRANSFER,
        A::SHADER_READ | A::SHADER_WRITE => PS::VERTEX_SHADER | PS::GEOMETRY_SHADER | PS::FRAGMENT_SHADER | PS::COMPUTE_SHADER,
        A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE => PS::EARLY_FRAGMENT_TESTS | PS::LATE_FRAGMENT_TESTS,
        A::HOST_READ | A::HOST_WRITE => PS::HOST,
        A::MEMORY_READ | A::MEMORY_WRITE => PS::empty(),
        A::INPUT_ATTACHMENT_READ => PS::FRAGMENT_SHADER,
    }
}

pub fn buffer_access_supported_pipeline_stages(access: BufferAccess) -> PipelineStage {
    type PS = PipelineStage;
    type A = BufferAccess;

    match access {
        A::TRANSFER_READ | A::TRANSFER_WRITE => PS::TRANSFER,
        A::INDEX_BUFFER_READ | A::VERTEX_BUFFER_READ => PS::VERTEX_INPUT,
        A::INDIRECT_COMMAND_READ => PS::DRAW_INDIRECT,
        A::CONSTANT_BUFFER_READ | A::SHADER_READ | A::SHADER_WRITE => PS::VERTEX_SHADER | PS::GEOMETRY_SHADER | PS::FRAGMENT_SHADER | PS::COMPUTE_SHADER,
        A::HOST_READ | A::HOST_WRITE => PS::HOST,
        A::MEMORY_READ | A::MEMORY_WRITE => PS::empty(),
    }
}
