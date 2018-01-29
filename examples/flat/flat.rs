
extern crate gfx_hal;
extern crate gfx_mem;
extern crate smallvec;
extern crate xfg;

use std::borrow::Borrow;

use gfx_hal::{Backend, Device, IndexType};
use gfx_hal::buffer::{IndexBufferView, Usage};
use gfx_hal::command::{CommandBuffer, RenderPassInlineEncoder, Primary};
use gfx_hal::device::ShaderError;
use gfx_hal::format::Format;
use gfx_hal::memory::{cast_slice, Pod, Properties};
use gfx_hal::pso::{DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType, DescriptorWrite, Element, ElemStride, EntryPoint, GraphicsShaderSet, ShaderStageFlags, VertexBufferSet};
use gfx_hal::queue::Transfer;
use gfx_mem::{Block, Factory, SmartAllocator, Type};
use smallvec::SmallVec;
use xfg::{DescriptorPool, Pass};

type Buffer<B> = <SmartAllocator<B> as Factory<B>>::Buffer;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct TrProjView {
    transform: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
}

unsafe impl Pod for TrProjView {}

struct Cache<B: Backend> {
    uniform: Buffer<B>,
    set: B::DescriptorSet,
}

struct Object<B: Backend> {
    indices: Buffer<B>,
    pos_color: Buffer<B>,
    index_count: u32,
    transform: [[f32; 4]; 4],
    cache: Option<Cache<B>>,
}

struct Camera {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
}

struct Scene<B: Backend> {
    objects: Vec<Object<B>>,
    camera: Camera,
    allocator: SmartAllocator<B>,
}

#[derive(Debug)]
struct DrawFlat;

impl<B> Pass<B, Scene<B>> for DrawFlat
where
    B: Backend,
{
    /// Name of the pass
    fn name(&self) -> &str {
        "DrawFlat"
    }

    /// Input attachments
    fn inputs(&self) -> usize { 0 }

    /// Color attachments
    fn colors(&self) -> usize { 1 }

    /// Uses depth attachment
    fn depth(&self) -> bool { true }

    /// Uses stencil attachment
    fn stencil(&self) -> bool { false }

    /// Vertices format
    fn vertices(&self) -> &[(&[Element<Format>], ElemStride)] {
        &[
            (
                &[
                    Element {
                        format: Format::Rgb32Float,
                        offset: 0,
                    },
                    Element {
                        format: Format::Rgba8Unorm,
                        offset: 12,
                    },
                ],
                28,
            )
        ]
    }

    fn bindings(&self) -> &[DescriptorSetLayoutBinding] {
        &[
            DescriptorSetLayoutBinding {
                binding: 0,
                ty: DescriptorType::UniformBuffer,
                count: 0,
                stage_flags: ShaderStageFlags::VERTEX,
            }
        ]
    }

    fn shaders<'a>(
        &self,
        shaders: &'a mut SmallVec<[B::ShaderModule; 5]>,
        device: &B::Device,
    ) -> Result<GraphicsShaderSet<'a, B>, ShaderError> {
        shaders.clear();
        shaders.push(device.create_shader_module(include_bytes!("vert.spv"))?);
        shaders.push(device.create_shader_module(include_bytes!("frag.spv"))?);

        Ok(GraphicsShaderSet {
            vertex: EntryPoint {
                entry: "main",
                module: &shaders[0],
                specialization: &[],
            },
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(EntryPoint {
                entry: "main",
                module: &shaders[1],
                specialization: &[],
            }),
        })
    }

    fn prepare<'a>(
        &mut self,
        pool: &mut DescriptorPool<B>,
        cbuf: &mut CommandBuffer<B, Transfer>,
        device: &B::Device,
        scene: &mut Scene<B>,
    )
    {
        let ref mut allocator = scene.allocator;
        // Update uniform cache
        for obj in &mut scene.objects {
            let trprojview = TrProjView {
                transform: obj.transform,
                projection: scene.camera.proj,
                view: scene.camera.view,
            };

            let cache = obj.cache.get_or_insert_with(|| {
                let buffer = allocator.create_buffer(device, (Type::General, Properties::DEVICE_LOCAL), ::std::mem::size_of::<TrProjView>() as u64, Usage::UNIFORM).unwrap();
                let set = pool.allocate(device);
                device.update_descriptor_sets(&[DescriptorSetWrite {
                    set: &set,
                    binding: 0,
                    array_offset: 0,
                    write: DescriptorWrite::UniformBuffer(vec![(buffer.borrow(), buffer.range())]),
                }]);
                Cache {
                    uniform: buffer,
                    set,
                }
            });
            cbuf.update_buffer(cache.uniform.borrow(), 0, cast_slice(&[trprojview]));
        }
    }

    fn draw_inline<'a>(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassInlineEncoder<B, Primary>,
        _device: &B::Device,
        scene: &Scene<B>,
    ) {       
        for object in &scene.objects {
            let cache = object.cache.as_ref().unwrap();
            encoder.bind_graphics_descriptor_sets(layout, 0, Some(&cache.set));
            encoder.bind_index_buffer(IndexBufferView {
                buffer: object.indices.borrow(),
                offset: 0,
                index_type: IndexType::U16,
            });
            encoder.bind_vertex_buffers(VertexBufferSet(vec![(object.pos_color.borrow(), 0)]));
            encoder.draw_indexed(
                0 .. object.index_count,
                0,
                0 .. 1,
            );
        }
    }

    fn cleanup(&mut self, pool: &mut DescriptorPool<B>, device: &B::Device, scene: &mut Scene<B>) {
        for object in &mut scene.objects {
            if let Some(cache) = object.cache.take() {
                pool.free(cache.set);
                scene.allocator.destroy_buffer(device, cache.uniform);
            }
        }
    }
}

fn main() {}