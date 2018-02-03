

// #![deny(unused_imports)]
#![deny(unused_must_use)]
#![allow(dead_code)]

extern crate xfg_examples;
extern crate smallvec;
use xfg_examples::*;

use std::borrow::Borrow;

use cgmath::{Deg, PerspectiveFov, SquareMatrix, Matrix4};

use gfx_hal::{Backend, Device, IndexType};
use gfx_hal::buffer::{IndexBufferView, Usage};
use gfx_hal::command::{ClearColor, ClearDepthStencil, CommandBuffer, RenderPassInlineEncoder, Primary};
use gfx_hal::device::ShaderError;
use gfx_hal::format::Format;
use gfx_hal::memory::{cast_slice, Pod};
use gfx_hal::pso::{DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType, DescriptorWrite, Element, ElemStride, EntryPoint, GraphicsShaderSet, ShaderStageFlags, VertexBufferSet};
use gfx_hal::queue::Transfer;
use gfx_mem::{Block, Factory, SmartAllocator};
use smallvec::SmallVec;
use xfg::{DescriptorPool, Pass, ColorAttachment, DepthStencilAttachment, GraphBuilder};

// use gfx_hal::pool::{CommandPool};


#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct TrProjView {
    transform: Matrix4<f32>,
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

unsafe impl Pod for TrProjView {}

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
                        format: Format::Rgba32Float,
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
                count: 1,
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
                let buffer = allocator.create_buffer(device, REQUEST_DEVICE_LOCAL, ::std::mem::size_of::<TrProjView>() as u64, Usage::UNIFORM).unwrap();
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
            encoder.bind_vertex_buffers(VertexBufferSet(vec![(object.vertices.borrow(), 0)]));
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

fn create_cube<B>(device: &B::Device, factory: &mut SmartAllocator<B>, transform: Matrix4<f32>) -> Object<B>
where
    B: Backend,
{
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct PosColor {
        position: [f32; 3],
        color: [f32; 4],
    }

    unsafe impl Pod for PosColor {}

    let vertices = vec![
        // Right
        PosColor {
            position: [0.5, -0.5, -0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, -0.5, 0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, 0.5, -0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, 0.5, 0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        // Left
        PosColor {
            position: [-0.5, -0.5, -0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PosColor {
            position: [-0.5, -0.5, 0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PosColor {
            position: [-0.5, 0.5, -0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PosColor {
            position: [-0.5, 0.5, 0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        // Top
        PosColor {
            position: [-0.5, 0.5, -0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        PosColor {
            position: [-0.5, 0.5, 0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, 0.5, -0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, 0.5, 0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        // Bottom
        PosColor {
            position: [-0.5, -0.5, -0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        PosColor {
            position: [-0.5, -0.5, 0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, -0.5, -0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        PosColor {
            position: [0.5, -0.5, 0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        // Front
        PosColor {
            position: [-0.5, -0.5, 0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        PosColor {
            position: [-0.5, 0.5, 0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        PosColor {
            position: [0.5, -0.5, 0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        PosColor {
            position: [0.5, 0.5, 0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        // Back
        PosColor {
            position: [-0.5, -0.5, -0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        PosColor {
            position: [-0.5, 0.5, -0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        PosColor {
            position: [0.5, -0.5, -0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
        PosColor {
            position: [0.5, 0.5, -0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
    ];

    let vertices: &[u8] = cast_slice(&vertices);

    let buffer = factory.create_buffer(device, REQUEST_CPU_VISIBLE, vertices.len() as u64, Usage::VERTEX).unwrap();
    {
        let start = buffer.range().start;
        let end = start + vertices.len() as u64;
        let mut writer = device.acquire_mapping_writer(buffer.memory(), start .. end).unwrap();
        writer.copy_from_slice(vertices);
        device.release_mapping_writer(writer);
    }

    let vertices = buffer;

    let indices: Vec<u16> = vec![
        // Left
        0, 1, 2,
        1, 2, 3,
        // Right
        4, 5, 6,
        5, 6, 7,
        // Top
        8, 9, 10,
        9, 10, 11,
        // Bottom
        12, 13, 14,
        13, 14, 15,
        // Front
        16, 17, 18,
        17, 18, 19,
        // Back
        20, 21, 22,
        21, 22, 23,
    ];

    let index_count = indices.len() as u32;

    let indices: &[u8] = cast_slice(&indices);

    let buffer = factory.create_buffer(device, REQUEST_CPU_VISIBLE, indices.len() as u64, Usage::INDEX).unwrap();
    {
        let mut writer = device.acquire_mapping_writer(buffer.memory(), buffer.range()).unwrap();
        writer.copy_from_slice(indices);
        device.release_mapping_writer(writer);
    }

    let indices = buffer;

    Object {
        vertices,
        indices,
        index_count,
        transform,
        cache: None,
    }
}

fn graph<'a, B>(surface_format: Format, colors: &'a mut Vec<ColorAttachment>, depths: &'a mut Vec<DepthStencilAttachment>) -> GraphBuilder<'a, B, Scene<B>>
where
    B: Backend,
{
    colors.push(ColorAttachment::new(surface_format).with_clear(ClearColor::Float([0.3, 0.4, 0.5, 1.0])));
    depths.push(DepthStencilAttachment::new(Format::D32Float).with_clear(ClearDepthStencil(1.0, 0)));

    let pass = DrawFlat.build()
        .with_color(0, colors.last().unwrap())
        .with_depth(depths.last().unwrap());

    GraphBuilder::new()
        .with_pass(pass)
        .with_present(colors.last().unwrap())
}

fn populate<B>(scene: &mut Scene<B>, device: &B::Device)
where
    B: Backend,
{
    let view = Matrix4::from_translation([0.0, 0.0, -5.0].into());
    let view = view * Matrix4::from_angle_x(Deg(-45.0));
    let view = view * Matrix4::from_angle_y(Deg(25.0));

    let proj: Matrix4<f32> = PerspectiveFov {
        fovy: Deg(60.0).into(),
        aspect: 1.0,
        near: 0.1,
        far: 2000.0,
    }.into();

    scene.objects = vec![create_cube(device, &mut scene.allocator, Matrix4::identity())];
    scene.camera = Camera {
        view,
        proj,
    };
}

fn main() {
    run(graph::<back::Backend>, populate);
}
