// #![deny(unused_imports)]
#![deny(unused_must_use)]
#![allow(dead_code)]

include!("../common/lib.rs");

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct TrProjView {
    transform: mat4,
    view: mat4,
    projection: mat4,
}

unsafe impl Pod for TrProjView {}

#[derive(Debug)]
struct DrawFlat<B: Backend> {
    pool: XfgDescriptorPool<B>,
}

impl<B> DrawFlat<B>
where
    B: Backend,
{
    fn vertices() -> Vec<(Vec<Element<Format>>, ElemStride)> {
        let vertices = vec![(
            vec![
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
        )];

        assert!(
            vertices[0]
                .0
                .iter()
                .eq(&*PosColor::VERTEX_FORMAT.attributes)
        );
        assert_eq!(vertices[0].1, PosColor::VERTEX_FORMAT.stride);

        vertices
    }

    fn bindings() -> &'static [DescriptorSetLayoutBinding] {
        &[DescriptorSetLayoutBinding {
            binding: 0,
            ty: DescriptorType::UniformBuffer,
            count: 1,
            stage_flags: ShaderStageFlags::VERTEX,
        }]
    }
}

impl<B> RenderPassDesc<B> for DrawFlat<B>
where
    B: Backend,
{
    fn name() -> &'static str {
        "DrawFlat"
    }

    fn colors() -> usize {
        1
    }

    fn depth() -> bool {
        true
    }

    fn layouts() -> Vec<Layout> {
        vec![Layout {
            sets: vec![SetLayout {
                bindings: Self::bindings().iter().cloned().collect(),
            }],
            push_constants: Vec::new(),
        }]
    }

    fn pipelines() -> Vec<Pipeline> {
        vec![Pipeline {
            layout: 0,
            vertices: Self::vertices(),
            colors: vec![ColorBlendDesc(ColorMask::ALL, BlendState::ALPHA)],
            depth_stencil: Some(DepthStencilDesc {
                depth: DepthTest::On {
                    fun: Comparison::LessEqual,
                    write: true,
                },
                depth_bounds: false,
                stencil: StencilTest::Off,
            }),
        }]
    }
}

impl<B> RenderPass<B, Factory<B>, Scene<B>> for DrawFlat<B>
where
    B: Backend,
{
    fn load_shader_sets<'a>(
        storage: &'a mut Vec<B::ShaderModule>,
        factory: &mut Factory<B>,
        aux: &mut Scene<B>,
    ) -> Vec<GraphicsShaderSet<'a, B>> {
        let offset = storage.len();
        storage.push(
            factory
                .create_shader_module(include_bytes!("vert.spv"))
                .unwrap(),
        );
        storage.push(
            factory
                .create_shader_module(include_bytes!("frag.spv"))
                .unwrap(),
        );

        vec![GraphicsShaderSet {
            vertex: EntryPoint {
                entry: "main",
                module: &storage[offset + 0],
                specialization: &[],
            },
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(EntryPoint {
                entry: "main",
                module: &storage[offset + 1],
                specialization: &[],
            }),
        }]
    }

    fn build<I>(_sampled: I, _storage: I, _device: &mut Factory<B>, _aux: &mut Scene<B>) -> Self {
        DrawFlat {
            pool: XfgDescriptorPool::new(),
        }
    }

    fn prepare<A, S>(
        &mut self,
        sets: &A,
        cbuf: &mut CommandBuffer<B, Graphics>,
        factory: &mut Factory<B>,
        scene: &Scene<B>,
    ) where
        A: Index<usize>,
        A::Output: Index<usize, Output = S>,
        S: Borrow<B::DescriptorSetLayout>,
    {
        use std::mem::size_of;

        let set = sets[0][0].borrow();

        let view = scene.camera.transform.inverse_transform().unwrap();

        // Update uniform cache
        for object in &scene.objects {
            let trprojview = TrProjView {
                transform: object.transform.into(),
                projection: scene.camera.projection.into(),
                view: view.into(),
            };

            let mut cache = unsafe { &mut *object.cache.get() };

            let cache = cache.get_or_insert_with(|| {
                let size = size_of::<TrProjView>() as u64;
                let buffer = factory
                    .create_buffer(
                        size,
                        Properties::DEVICE_LOCAL,
                        buffer::Usage::UNIFORM | buffer::Usage::TRANSFER_DST,
                    )
                    .unwrap();
                let set = self.pool.allocate(factory, set, Self::bindings());
                factory.write_descriptor_sets(Some(DescriptorSetWrite {
                    set: &set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(Descriptor::Buffer(buffer.borrow(), Some(0)..Some(size))),
                }));
                Cache {
                    uniforms: vec![buffer],
                    set,
                }
            });

            cbuf.update_buffer(cache.uniforms[0].borrow(), 0, cast_slice(&[trprojview]));

            cbuf.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::VERTEX_SHADER,
                Dependencies::empty(),
                Some(Barrier::Buffer {
                    target: cache.uniforms[0].borrow(),
                    states: buffer::Access::TRANSFER_WRITE..buffer::Access::SHADER_READ,
                }),
            );
        }
    }

    fn draw<L, P>(
        &mut self,
        layouts: &L,
        pipelines: &P,
        mut encoder: RenderPassInlineEncoder<B, Primary>,
        scene: &Scene<B>,
    ) where
        L: Index<usize>,
        L::Output: Borrow<B::PipelineLayout>,
        P: Index<usize>,
        P::Output: Borrow<B::GraphicsPipeline>,
    {
        let pipeline = pipelines[0].borrow();
        encoder.bind_graphics_pipeline(pipeline);
        let layout = layouts[0].borrow();
        for object in &scene.objects {
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                Some(&unsafe { &*object.cache.get() }.as_ref().unwrap().set),
            );
            let mut vbs = VertexBufferSet(Vec::new());
            let bind = object
                .mesh
                .bind(&[PosColor::VERTEX_FORMAT], &mut vbs)
                .unwrap();

            bind.draw(vbs, &mut encoder);
        }
    }

    fn dispose(mut self, factory: &mut Factory<B>, scene: &mut Scene<B>) {
        for object in &mut scene.objects {
            for cache in unsafe { &mut *object.cache.get() }.take() {
                self.pool.free(cache.set);
                for uniform in cache.uniforms {
                    factory.destroy_buffer(uniform);
                }
            }
        }
    }
}

fn create_cube<B>(factory: &mut Factory<B>) -> Mesh<B>
where
    B: Backend,
{
    let vertices = [
        // Right
        PosColor {
            position: [0.5, -0.5, -0.5].into(),
            color: [1.0, 0.0, 0.0, 1.0].into(),
        },
        PosColor {
            position: [0.5, -0.5, 0.5].into(),
            color: [1.0, 0.0, 0.0, 1.0].into(),
        },
        PosColor {
            position: [0.5, 0.5, -0.5].into(),
            color: [1.0, 0.0, 0.0, 1.0].into(),
        },
        PosColor {
            position: [0.5, 0.5, 0.5].into(),
            color: [1.0, 0.0, 0.0, 1.0].into(),
        },
        // Left
        PosColor {
            position: [-0.5, -0.5, -0.5].into(),
            color: [1.0, 0.0, 0.0, 1.0].into(),
        },
        PosColor {
            position: [-0.5, -0.5, 0.5].into(),
            color: [1.0, 0.0, 0.0, 1.0].into(),
        },
        PosColor {
            position: [-0.5, 0.5, -0.5].into(),
            color: [1.0, 0.0, 0.0, 1.0].into(),
        },
        PosColor {
            position: [-0.5, 0.5, 0.5].into(),
            color: [1.0, 0.0, 0.0, 1.0].into(),
        },
        // Top
        PosColor {
            position: [-0.5, 0.5, -0.5].into(),
            color: [0.0, 1.0, 0.0, 1.0].into(),
        },
        PosColor {
            position: [-0.5, 0.5, 0.5].into(),
            color: [0.0, 1.0, 0.0, 1.0].into(),
        },
        PosColor {
            position: [0.5, 0.5, -0.5].into(),
            color: [0.0, 1.0, 0.0, 1.0].into(),
        },
        PosColor {
            position: [0.5, 0.5, 0.5].into(),
            color: [0.0, 1.0, 0.0, 1.0].into(),
        },
        // Bottom
        PosColor {
            position: [-0.5, -0.5, -0.5].into(),
            color: [0.0, 1.0, 0.0, 1.0].into(),
        },
        PosColor {
            position: [-0.5, -0.5, 0.5].into(),
            color: [0.0, 1.0, 0.0, 1.0].into(),
        },
        PosColor {
            position: [0.5, -0.5, -0.5].into(),
            color: [0.0, 1.0, 0.0, 1.0].into(),
        },
        PosColor {
            position: [0.5, -0.5, 0.5].into(),
            color: [0.0, 1.0, 0.0, 1.0].into(),
        },
        // Front
        PosColor {
            position: [-0.5, -0.5, 0.5].into(),
            color: [0.0, 0.0, 1.0, 1.0].into(),
        },
        PosColor {
            position: [-0.5, 0.5, 0.5].into(),
            color: [0.0, 0.0, 1.0, 1.0].into(),
        },
        PosColor {
            position: [0.5, -0.5, 0.5].into(),
            color: [0.0, 0.0, 1.0, 1.0].into(),
        },
        PosColor {
            position: [0.5, 0.5, 0.5].into(),
            color: [0.0, 0.0, 1.0, 1.0].into(),
        },
        // Back
        PosColor {
            position: [-0.5, -0.5, -0.5].into(),
            color: [0.0, 0.0, 1.0, 1.0].into(),
        },
        PosColor {
            position: [-0.5, 0.5, -0.5].into(),
            color: [0.0, 0.0, 1.0, 1.0].into(),
        },
        PosColor {
            position: [0.5, -0.5, -0.5].into(),
            color: [0.0, 0.0, 1.0, 1.0].into(),
        },
        PosColor {
            position: [0.5, 0.5, -0.5].into(),
            color: [0.0, 0.0, 1.0, 1.0].into(),
        },
    ];

    let indices = [
        // Left
        0u32,
        1,
        2,
        1,
        2,
        3,
        // Right
        4,
        5,
        6,
        5,
        6,
        7,
        // Top
        8,
        9,
        10,
        9,
        10,
        11,
        // Bottom
        12,
        13,
        14,
        13,
        14,
        15,
        // Front
        16,
        17,
        18,
        17,
        18,
        19,
        // Back
        20,
        21,
        22,
        21,
        22,
        23,
    ];

    let builder = MeshBuilder::new()
        .with_vertices(&vertices[..])
        .with_indices(&indices[..]);

    builder.build(QueueFamilyId(0), factory).unwrap()
}

fn graph<B>(kind: image::Kind, surface_format: Format, graph: &mut XfgGraphBuilder<B>) -> ImageId
where
    B: Backend,
{
    let surface = graph.create_image(
        kind,
        surface_format,
        Some(ClearValue::Color(ClearColor::Float([
            0.05, 0.02, 0.05, 1.0,
        ]))),
    );
    let depth = graph.create_image(
        kind,
        Format::D32Float,
        Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
    );

    graph.add_node(DrawFlat::builder().with_image(surface).with_image(depth));
    surface
}

fn fill<B>(scene: &mut Scene<B>, factory: &mut Factory<B>)
where
    B: Backend,
{
    let transform = Matrix4::from_translation([0.0, 0.0, 5.0].into());
    let transform = Matrix4::from_angle_x(Deg(45.0)) * transform;
    let transform = Matrix4::from_angle_y(Deg(25.0)) * transform;

    scene.camera.transform = transform;

    let cube = create_cube(factory);

    scene.objects = vec![Object {
        mesh: Arc::new(cube),
        transform: Matrix4::one(),
        data: (),
        cache: UnsafeCell::new(None),
    }];
}

fn main() {
    run(graph, fill);
}
