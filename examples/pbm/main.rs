#![deny(unused_must_use)]
#![allow(unused_imports)]
#![allow(dead_code)]

include!("../common/lib.rs");

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Material {
    albedo: [f32; 3],
    metallic: f32,
    emission: [f32; 3],
    roughness: f32,
    ambient_occlusion: f32,
}

#[derive(Debug)]
struct DrawColorDepthNormal<B: Backend> {
    pool: XfgDescriptorPool<B>,
}

impl<B> DrawColorDepthNormal<B>
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
                    format: Format::Rgb32Float,
                    offset: 12,
                },
            ],
            24,
        )];

        assert!(vertices[0].0.iter().eq(&*PosNorm::VERTEX_FORMAT.attributes));

        vertices
    }

    fn bindings() -> &'static [DescriptorSetLayoutBinding] {
        &[
            DescriptorSetLayoutBinding {
                binding: 0,
                ty: DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: ShaderStageFlags::VERTEX,
                immutable_samplers: false,
            },
            DescriptorSetLayoutBinding {
                binding: 1,
                ty: DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
                immutable_samplers: false,
            },
        ]
    }
}

impl<B> RenderPassDesc<B> for DrawColorDepthNormal<B>
where
    B: Backend,
{
    fn name() -> &'static str {
        "DrawColorDepthNormal"
    }

    fn colors() -> usize {
        4
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
            colors: vec![ColorBlendDesc(ColorMask::ALL, BlendState::Off); 4],
            depth_stencil: DepthStencilDesc {
                depth: DepthTest::On {
                    fun: Comparison::LessEqual,
                    write: true,
                },
                depth_bounds: false,
                stencil: StencilTest::Off,
            },
        }]
    }
}

impl<B> RenderPass<B, Factory<B>, Scene<B, Material>> for DrawColorDepthNormal<B>
where
    B: Backend,
{
    fn load_shader_sets<'a>(
        storage: &'a mut Vec<B::ShaderModule>,
        factory: &mut Factory<B>,
        _aux: &mut Scene<B, Material>,
    ) -> Vec<GraphicsShaderSet<'a, B>> {
        let offset = storage.len();
        storage.push(
            factory
                .create_shader_module(include_bytes!("first.vert.spv"))
                .unwrap(),
        );
        storage.push(
            factory
                .create_shader_module(include_bytes!("first.frag.spv"))
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

    fn build<I>(
        _sampled: I,
        _storage: I,
        _device: &mut Factory<B>,
        _aux: &mut Scene<B, Material>,
    ) -> Self {
        DrawColorDepthNormal {
            pool: XfgDescriptorPool::new(),
        }
    }

    #[inline]
    fn prepare<A, S>(
        &mut self,
        sets: &A,
        cbuf: &mut CommandBuffer<B, Graphics>,
        factory: &mut Factory<B>,
        scene: &Scene<B, Material>,
    ) where
        A: Index<usize>,
        A::Output: Index<usize, Output = S>,
        S: Borrow<B::DescriptorSetLayout>,
    {
        use std::iter::once;

        let set = sets[0][0].borrow();

        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct VertexArgs {
            proj: Matrix4<f32>,
            view: Matrix4<f32>,
            model: Matrix4<f32>,
        }

        unsafe impl Pod for VertexArgs {}

        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct FragmentArgs {
            albedo: [f32; 3],
            roughness: f32,
            emission: [f32; 3],
            metallic: f32,
            ambient_occlusion: f32,
        }

        unsafe impl Pod for FragmentArgs {}

        profile!("DrawColorDepthNormal::prepare");
        let view = scene.camera.transform.inverse_transform().unwrap();
        // Update uniform cache
        for object in &scene.objects {
            profile!("Prepare object");
            let mut cache = unsafe { &mut *object.cache.get() };
            cache.get_or_insert_with(|| {
                profile!("Build cache");

                let vertex_args = VertexArgs {
                    model: object.transform,
                    proj: scene.camera.projection,
                    view,
                };

                let fragment_args = FragmentArgs {
                    albedo: object.data.albedo,
                    metallic: object.data.metallic,
                    emission: object.data.emission,
                    roughness: object.data.roughness,
                    ambient_occlusion: object.data.ambient_occlusion,
                };

                let vertex_args_range = Some(0)..Some(::std::mem::size_of::<VertexArgs>() as u64);
                let fragment_args_offset = shift_for_alignment(256, vertex_args_range.end.unwrap());
                let fragment_args_range = Some(fragment_args_offset)
                    ..Some(fragment_args_offset + ::std::mem::size_of::<FragmentArgs>() as u64);
                let buffer = factory
                    .create_buffer(
                        fragment_args_range.end.unwrap(),
                        Properties::DEVICE_LOCAL,
                        buffer::Usage::UNIFORM | buffer::Usage::TRANSFER_DST,
                    )
                    .unwrap();
                let set = self.pool.allocate(factory, set, Self::bindings());
                factory.write_descriptor_sets(
                    once(DescriptorSetWrite {
                        set: &set,
                        binding: 0,
                        array_offset: 0,
                        descriptors: Some(Descriptor::Buffer(
                            buffer.borrow(),
                            vertex_args_range.clone(),
                        )),
                    }).chain(once(DescriptorSetWrite {
                        set: &set,
                        binding: 1,
                        array_offset: 0,
                        descriptors: Some(Descriptor::Buffer(
                            buffer.borrow(),
                            fragment_args_range.clone(),
                        )),
                    })),
                );

                cbuf.update_buffer(
                    buffer.borrow(),
                    vertex_args_range.start.unwrap(),
                    cast_slice(&[vertex_args]),
                );
                cbuf.update_buffer(
                    buffer.borrow(),
                    fragment_args_range.start.unwrap(),
                    cast_slice(&[fragment_args]),
                );
                cbuf.pipeline_barrier(
                    PipelineStage::TRANSFER
                        ..PipelineStage::VERTEX_SHADER | PipelineStage::FRAGMENT_SHADER,
                    Dependencies::empty(),
                    Some(Barrier::Buffer {
                        target: buffer.borrow(),
                        states: buffer::Access::TRANSFER_WRITE..buffer::Access::SHADER_READ,
                    }),
                );
                Cache {
                    uniforms: vec![buffer],
                    set,
                }
            });
        }
    }

    fn draw<L, P>(
        &mut self,
        layouts: &L,
        pipelines: &P,
        mut encoder: RenderPassInlineEncoder<B, Primary>,
        scene: &Scene<B, Material>,
    ) where
        L: Index<usize>,
        L::Output: Borrow<B::PipelineLayout>,
        P: Index<usize>,
        P::Output: Borrow<B::GraphicsPipeline>,
    {
        let pipeline = pipelines[0].borrow();
        encoder.bind_graphics_pipeline(pipeline);
        let layout = layouts[0].borrow();

        profile!("DrawColorDepthNormal::draw");
        for object in &scene.objects {
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                Some(&unsafe { &*object.cache.get() }.as_ref().unwrap().set),
                empty::<DescriptorSetOffset>(),
            );
            let mut vbs = VertexBufferSet(Vec::new());
            let bind = object
                .mesh
                .bind(&[PosNorm::VERTEX_FORMAT], &mut vbs)
                .unwrap();

            bind.draw(vbs, &mut encoder);
        }
    }

    fn dispose(mut self, factory: &mut Factory<B>, scene: &mut Scene<B, Material>) {
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

#[derive(Debug)]
struct DrawLights<B: Backend> {
    pool: XfgDescriptorPool<B>,
    input: [*const B::ImageView; 4],
}

unsafe impl<B: Backend> Send for DrawLights<B> {}
unsafe impl<B: Backend> Sync for DrawLights<B> {}

impl<B> DrawLights<B>
where
    B: Backend,
{
    fn bindings() -> &'static [DescriptorSetLayoutBinding] {
        &[
            DescriptorSetLayoutBinding {
                binding: 0,
                ty: DescriptorType::StorageImage,
                count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
                immutable_samplers: false,
            },
            DescriptorSetLayoutBinding {
                binding: 1,
                ty: DescriptorType::StorageImage,
                count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
                immutable_samplers: false,
            },
            DescriptorSetLayoutBinding {
                binding: 2,
                ty: DescriptorType::StorageImage,
                count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
                immutable_samplers: false,
            },
            DescriptorSetLayoutBinding {
                binding: 3,
                ty: DescriptorType::StorageImage,
                count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
                immutable_samplers: false,
            },
            DescriptorSetLayoutBinding {
                binding: 4,
                ty: DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
                immutable_samplers: false,
            },
        ]
    }
}

impl<B> RenderPassDesc<B> for DrawLights<B>
where
    B: Backend,
{
    fn name() -> &'static str {
        "DrawLights"
    }

    fn storage() -> usize {
        4
    }

    fn colors() -> usize {
        1
    }

    fn depth() -> bool {
        false
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
            vertices: Vec::new(),
            colors: vec![ColorBlendDesc(ColorMask::ALL, BlendState::ADD)],
            depth_stencil: DepthStencilDesc {
                depth: DepthTest::On {
                    fun: Comparison::LessEqual,
                    write: true,
                },
                depth_bounds: false,
                stencil: StencilTest::Off,
            },
        }]
    }
}

impl<B, T> RenderPass<B, Factory<B>, Scene<B, T>> for DrawLights<B>
where
    B: Backend,
{
    fn load_shader_sets<'a>(
        storage: &'a mut Vec<B::ShaderModule>,
        factory: &mut Factory<B>,
        _aux: &mut Scene<B, T>,
    ) -> Vec<GraphicsShaderSet<'a, B>> {
        let offset = storage.len();
        storage.push(
            factory
                .create_shader_module(include_bytes!("second.vert.spv"))
                .unwrap(),
        );
        storage.push(
            factory
                .create_shader_module(include_bytes!("second.frag.spv"))
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

    fn build<I>(_sampled: I, storage: I, _factory: &mut Factory<B>, _aux: &mut Scene<B, T>) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<B::ImageView>,
    {
        let mut inputs = storage.into_iter();
        DrawLights {
            pool: XfgDescriptorPool::new(),
            input: [
                inputs.next().unwrap().borrow(),
                inputs.next().unwrap().borrow(),
                inputs.next().unwrap().borrow(),
                inputs.next().unwrap().borrow(),
            ],
        }
    }

    fn prepare<A, S>(
        &mut self,
        sets: &A,
        cbuf: &mut CommandBuffer<B, Graphics>,
        factory: &mut Factory<B>,
        scene: &Scene<B, T>,
    ) where
        A: Index<usize>,
        A::Output: Index<usize, Output = S>,
        S: Borrow<B::DescriptorSetLayout>,
    {
        use std::iter::once;

        let set = sets[0][0].borrow();

        #[derive(Clone, Copy, Debug, PartialEq)]
        struct FragmentArgs {
            light_position: [f32; 3],
            _pad0: f32,
            color: [f32; 3],
            _pad1: f32,
            camera_position: [f32; 3],
            _pad2: f32,
            ambient_light: [f32; 3],
            _pad3: f32,
        }

        unsafe impl Pod for FragmentArgs {}

        let size = ::std::mem::size_of::<FragmentArgs>() as u64;

        let camera_position = scene
            .camera
            .transform
            .transform_point(Point3::origin())
            .into();

        // Update uniform cache
        for light in &scene.lights {
            let fragment_args = FragmentArgs {
                light_position: light.transform.transform_point(Point3::origin()).into(),
                color: light.color,
                camera_position,
                ambient_light: scene.ambient.0,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
            };

            let mut cache = unsafe { &mut *light.cache.get() };

            cache.get_or_insert_with(|| {
                let buffer = factory
                    .create_buffer(
                        size,
                        Properties::DEVICE_LOCAL,
                        buffer::Usage::UNIFORM | buffer::Usage::TRANSFER_DST,
                    )
                    .unwrap();
                let set = self.pool.allocate(factory, set, Self::bindings());
                factory.write_descriptor_sets(
                    once(DescriptorSetWrite {
                        set: &set,
                        binding: 0,
                        array_offset: 0,
                        descriptors: Some(Descriptor::Image(
                            unsafe { &*self.input[0] },
                            image::Layout::General,
                        )),
                    }).chain(once(DescriptorSetWrite {
                        set: &set,
                        binding: 1,
                        array_offset: 0,
                        descriptors: Some(Descriptor::Image(
                            unsafe { &*self.input[1] },
                            image::Layout::General,
                        )),
                    }))
                        .chain(once(DescriptorSetWrite {
                            set: &set,
                            binding: 2,
                            array_offset: 0,
                            descriptors: Some(Descriptor::Image(
                                unsafe { &*self.input[2] },
                                image::Layout::General,
                            )),
                        }))
                        .chain(once(DescriptorSetWrite {
                            set: &set,
                            binding: 3,
                            array_offset: 0,
                            descriptors: Some(Descriptor::Image(
                                unsafe { &*self.input[3] },
                                image::Layout::General,
                            )),
                        }))
                        .chain(once(DescriptorSetWrite {
                            set: &set,
                            binding: 4,
                            array_offset: 0,
                            descriptors: Some(Descriptor::Buffer(
                                buffer.borrow(),
                                Some(0)..Some(size),
                            )),
                        })),
                );

                cbuf.update_buffer(buffer.borrow(), 0, cast_slice(&[fragment_args]));

                cbuf.pipeline_barrier(
                    PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                    Dependencies::empty(),
                    Some(Barrier::Buffer {
                        target: buffer.borrow(),
                        states: buffer::Access::TRANSFER_WRITE..buffer::Access::SHADER_READ,
                    }),
                );

                Cache {
                    uniforms: vec![buffer],
                    set,
                }
            });
        }
    }

    fn draw<L, P>(
        &mut self,
        layouts: &L,
        pipelines: &P,
        mut encoder: RenderPassInlineEncoder<B, Primary>,
        scene: &Scene<B, T>,
    ) where
        L: Index<usize>,
        L::Output: Borrow<B::PipelineLayout>,
        P: Index<usize>,
        P::Output: Borrow<B::GraphicsPipeline>,
    {
        let pipeline = pipelines[0].borrow();
        encoder.bind_graphics_pipeline(pipeline);
        let layout = layouts[0].borrow();

        for light in &scene.lights {
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                Some(&unsafe { &*light.cache.get() }.as_ref().unwrap().set),
                empty::<DescriptorSetOffset>(),
            );
            encoder.draw(0..3, 0..1);
        }
    }

    fn dispose(mut self, factory: &mut Factory<B>, scene: &mut Scene<B, T>) {
        for light in &mut scene.lights {
            for cache in unsafe { &mut *light.cache.get() }.take() {
                self.pool.free(cache.set);
                for uniform in cache.uniforms {
                    factory.destroy_buffer(uniform);
                }
            }
        }
    }
}

fn graph<B>(
    kind: image::Kind,
    surface_format: Format,
    graph: &mut XfgGraphBuilder<B, Material>,
) -> ImageId
where
    B: Backend,
{
    let ambient_roughness = graph.create_image(
        kind,
        surface_format,
        Some(ClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 1.0]))),
    );
    let emission_metallic = graph.create_image(
        kind,
        surface_format,
        Some(ClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 1.0]))),
    );
    let normal_normal_ambient_occlusion = graph.create_image(
        kind,
        Format::Rgba32Float,
        Some(ClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 1.0]))),
    );
    let position_depth = graph.create_image(
        kind,
        Format::Rgba32Float,
        Some(ClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 1.0]))),
    );
    let depth = graph.create_image(
        kind,
        Format::D32Float,
        Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
    );

    let shaded = graph.create_image(
        kind,
        surface_format,
        Some(ClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 1.0]))),
    );

    let first = graph.add_node(
        DrawColorDepthNormal::builder()
            .with_image(ambient_roughness)
            .with_image(emission_metallic)
            .with_image(normal_normal_ambient_occlusion)
            .with_image(position_depth)
            .with_image(depth),
    );

    graph.add_node(
        DrawLights::builder()
            .with_image(ambient_roughness)
            .with_image(emission_metallic)
            .with_image(normal_normal_ambient_occlusion)
            .with_image(position_depth)
            .with_image(shaded)
            .with_dependency(first),
    );

    shaded
}

fn fill<B>(scene: &mut Scene<B, Material>, factory: &mut Factory<B>)
where
    B: Backend,
{
    scene.camera.transform = Matrix4::from_translation([0.0, 0.0, 15.0].into());

    let mut data = Material {
        albedo: [1.0; 3],
        metallic: 0.0,
        emission: [0.0, 0.0, 0.0],
        roughness: 0.0,
        ambient_occlusion: 1.0,
    };

    let sphere = Arc::new(create_sphere(factory));

    for i in 0..6 {
        for j in 0..6 {
            let transform = Matrix4::from_translation(
                [2.5 * (i as f32) - 6.25, 2.5 * (j as f32) - 6.25, 0.0].into(),
            );
            data.metallic = j as f32 * 0.2;
            data.roughness = i as f32 * 0.2;
            scene.objects.push(Object {
                mesh: sphere.clone(),
                data,
                transform,
                cache: UnsafeCell::new(None),
            });
        }
    }

    scene.lights.push(Light {
        color: [0.0, 0.623529411764706, 0.419607843137255],
        transform: Matrix4::from_translation([-6.25, -6.25, 10.0].into()),
        cache: UnsafeCell::new(None),
    });

    scene.lights.push(Light {
        color: [0.768627450980392, 0.007843137254902, 0.2],
        transform: Matrix4::from_translation([6.25, -6.25, 10.0].into()),
        cache: UnsafeCell::new(None),
    });

    scene.lights.push(Light {
        color: [1.0, 0.827450980392157, 0.0],
        transform: Matrix4::from_translation([-6.25, 6.25, 10.0].into()),
        cache: UnsafeCell::new(None),
    });

    scene.lights.push(Light {
        color: [0.0, 0.529411764705882, 0.741176470588235],
        transform: Matrix4::from_translation([6.25, 6.25, 10.0].into()),
        cache: UnsafeCell::new(None),
    });
}

fn main() {
    run(graph, fill);
}

fn create_sphere<B>(factory: &mut Factory<B>) -> Mesh<B>
where
    B: Backend,
{
    use genmesh::generators::{IndexedPolygon, SharedVertex, SphereUV};
    use genmesh::{EmitTriangles, Triangle};

    let sphere = SphereUV::new(40, 20);

    let vertices = sphere
        .shared_vertex_iter()
        .map(|v| PosNorm {
            position: v.pos.into(),
            normal: v.normal.into(),
        })
        .collect::<Vec<_>>();

    let indices = sphere
        .indexed_polygon_iter()
        .flat_map(|polygon| {
            let mut indices = SmallVec::<[u16; 6]>::new();
            polygon.emit_triangles(|Triangle { x, y, z }| {
                indices.push(x as u16);
                indices.push(y as u16);
                indices.push(z as u16);
            });
            indices
        })
        .collect::<Vec<_>>();

    let builder = MeshBuilder::new()
        .with_vertices(&vertices[..])
        .with_indices(&indices[..]);

    builder.build(QueueFamilyId(0), factory).unwrap()
}

fn shift_for_alignment<T>(alignment: T, offset: T) -> T
where
    T: From<u8>
        + ::std::ops::Add<Output = T>
        + ::std::ops::Sub<Output = T>
        + ::std::ops::BitOr<Output = T>
        + PartialOrd,
{
    if offset > 0.into() && alignment > 0.into() {
        ((offset - 1.into()) | (alignment - 1.into())) + 1.into()
    } else {
        offset
    }
}
