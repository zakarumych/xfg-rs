// #![deny(unused_imports)]
#![deny(unused_must_use)]
#![allow(dead_code)]

extern crate smallvec;
extern crate xfg_examples;
use xfg_examples::*;

use std::borrow::Borrow;
use std::sync::Arc;

use cgmath::{Deg, Matrix4, Transform};

use gfx_hal::{Backend, Device, IndexType};
use gfx_hal::buffer::{IndexBufferView, Usage};
use gfx_hal::command::{ClearColor, ClearDepthStencil, CommandBuffer, Primary,
                       RenderPassInlineEncoder};
use gfx_hal::device::ShaderError;
use gfx_hal::format::Format;
use gfx_hal::memory::{cast_slice, Pod};
use gfx_hal::pso::{DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType,
                   Descriptor, ElemStride, Element, EntryPoint, GraphicsShaderSet,
                   ShaderStageFlags, VertexBufferSet};
use gfx_hal::queue::Transfer;
use mem::{Block, Factory, SmartAllocator};
use smallvec::SmallVec;
use xfg::{ColorAttachment, DepthStencilAttachment, DescriptorPool, GraphBuilder, Pass, PassDesc,
          PassShaders};

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

impl PassDesc for DrawFlat {
    fn name(&self) -> &str {
        "DrawFlat"
    }

    fn sampled(&self) -> usize {
        0
    }

    fn storage(&self) -> usize {
        0
    }

    fn inputs(&self) -> usize {
        0
    }

    fn colors(&self) -> usize {
        1
    }

    fn depth(&self) -> bool {
        true
    }

    fn stencil(&self) -> bool {
        false
    }

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
            ),
        ]
    }

    fn bindings(&self) -> &[DescriptorSetLayoutBinding] {
        &[
            DescriptorSetLayoutBinding {
                binding: 0,
                ty: DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: ShaderStageFlags::VERTEX,
            },
        ]
    }
}

impl<B> PassShaders<B> for DrawFlat
where
    B: Backend,
{
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
}

impl<B> Pass<B, Scene<B>> for DrawFlat
where
    B: Backend,
{
    fn prepare<'a>(
        &mut self,
        pool: &mut DescriptorPool<B>,
        cbuf: &mut CommandBuffer<B, Transfer>,
        device: &B::Device,
        _inputs: &[&B::Image],
        frame: usize,
        scene: &mut Scene<B>,
    ) {
        let ref mut allocator = scene.allocator;
        let view = scene.camera.transform.inverse_transform().unwrap();

        // Update uniform cache
        for obj in &mut scene.objects {
            let trprojview = TrProjView {
                transform: obj.transform,
                projection: scene.camera.projection,
                view,
            };

            let grow = (obj.cache.len()..frame + 1).map(|_| None);
            obj.cache.extend(grow);

            let cache = obj.cache[frame].get_or_insert_with(|| {
                let size = ::std::mem::size_of::<TrProjView>() as u64;
                let buffer = allocator
                    .create_buffer(
                        device,
                        REQUEST_DEVICE_LOCAL,
                        size,
                        Usage::UNIFORM | Usage::TRANSFER_DST,
                    )
                    .unwrap();
                let set = pool.allocate(device);
                device.write_descriptor_sets(Some(
                    DescriptorSetWrite {
                        set: &set,
                        binding: 0,
                        array_offset: 0,
                        descriptors: Some(Descriptor::Buffer(buffer.borrow(), Some(0)..Some(size))),
                    },
                ));
                Cache {
                    uniforms: vec![buffer],
                    views: Vec::new(),
                    set,
                }
            });
            cbuf.update_buffer(cache.uniforms[0].borrow(), 0, cast_slice(&[trprojview]));
        }
    }

    fn draw_inline<'a>(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassInlineEncoder<B, Primary>,
        _device: &B::Device,
        _inputs: &[&B::Image],
        frame: usize,
        scene: &Scene<B>,
    ) {
        for object in &scene.objects {
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                Some(&object.cache[frame].as_ref().unwrap().set),
            );
            encoder.bind_index_buffer(IndexBufferView {
                buffer: object.mesh.indices.borrow(),
                offset: 0,
                index_type: IndexType::U16,
            });
            encoder.bind_vertex_buffers(VertexBufferSet(vec![(object.mesh.vertices.borrow(), 0)]));
            encoder.draw_indexed(0..object.mesh.index_count, 0, 0..1);
        }
    }

    fn cleanup(&mut self, pool: &mut DescriptorPool<B>, device: &B::Device, scene: &mut Scene<B>) {
        for object in &mut scene.objects {
            for cache in object.cache.drain(..) {
                if let Some(cache) = cache {
                    pool.free(cache.set);
                    for uniform in cache.uniforms {
                        scene.allocator.destroy_buffer(device, uniform);
                    }
                }
            }
        }
    }
}

fn create_cube<B>(device: &B::Device, factory: &mut SmartAllocator<B>) -> Mesh<B>
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

    let buffer = factory
        .create_buffer(
            device,
            REQUEST_CPU_VISIBLE,
            vertices.len() as u64,
            Usage::VERTEX,
        )
        .unwrap();
    {
        let start = buffer.range().start;
        let end = start + vertices.len() as u64;
        let mut writer = device
            .acquire_mapping_writer(buffer.memory(), start..end)
            .unwrap();
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

    let buffer = factory
        .create_buffer(
            device,
            REQUEST_CPU_VISIBLE,
            indices.len() as u64,
            Usage::INDEX,
        )
        .unwrap();
    {
        let start = buffer.range().start;
        let end = start + indices.len() as u64;
        let mut writer = device
            .acquire_mapping_writer(buffer.memory(), start..end)
            .unwrap();
        writer.copy_from_slice(indices);
        device.release_mapping_writer(writer);
    }

    let indices = buffer;

    Mesh {
        vertices,
        indices,
        index_count,
    }
}

fn graph(surface_format: Format, graph: &mut GraphBuilder<DrawFlat>) {
    let color = graph.add_attachment(
        ColorAttachment::new(surface_format).with_clear(ClearColor::Float([0.3, 0.4, 0.5, 1.0])),
    );
    let depth = graph.add_attachment(
        DepthStencilAttachment::new(Format::D32Float).with_clear(ClearDepthStencil(1.0, 0)),
    );

    let pass = DrawFlat.build()
        .with_color(color)
        .with_depth_stencil(depth)
        ;

    graph.add_pass(pass).set_present(color);
}

fn fill<B>(scene: &mut Scene<B>, device: &B::Device)
where
    B: Backend,
{
    let transform = Matrix4::from_translation([0.0, 0.0, 5.0].into());
    let transform = Matrix4::from_angle_x(Deg(45.0)) * transform;
    let transform = Matrix4::from_angle_y(Deg(25.0)) * transform;

    scene.camera.transform = transform;

    let cube = create_cube(device, &mut scene.allocator);

    scene.objects = vec![
        Object {
            mesh: Arc::new(cube),
            transform: Matrix4::one(),
            data: (),
            cache: Vec::new(),
        },
    ];
}

fn main() {
    run(graph, fill);
}
