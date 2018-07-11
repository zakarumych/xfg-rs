#![deny(unused_must_use)]
#![allow(unused_imports)]
#![allow(dead_code)]

#[macro_use]
extern crate conrod;

extern crate rusttype;

include!("../common/lib.rs");

use conrod::{color::Rgba, render::PrimitiveKind, text::GlyphCache, Ui};

/// Vertex format with position and RGBA8 color attributes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosColorTex {
    pub position: Position,
    pub color: Color,
    pub tex: TexCoord,
}

unsafe impl Pod for PosColorTex {}

impl AsVertexFormat for PosColorTex {
    const VERTEX_FORMAT: VertexFormat<'static> = VertexFormat {
        attributes: Cow::Borrowed(&[
            <Self as WithAttribute<Position>>::ELEMENT,
            <Self as WithAttribute<Color>>::ELEMENT,
            <Self as WithAttribute<TexCoord>>::ELEMENT,
        ]),
        stride: Position::SIZE + Color::SIZE + TexCoord::SIZE,
    };
}

impl WithAttribute<Position> for PosColorTex {
    const ELEMENT: Element<Format> = Element {
        offset: 0,
        format: Position::SELF,
    };
}

impl WithAttribute<Color> for PosColorTex {
    const ELEMENT: Element<Format> = Element {
        offset: Position::SIZE,
        format: Color::SELF,
    };
}

impl WithAttribute<TexCoord> for PosColorTex {
    const ELEMENT: Element<Format> = Element {
        offset: Position::SIZE + Color::SIZE,
        format: TexCoord::SELF,
    };
}

struct Vertex<B: Backend> {
    buffer: Buffer<B>,
    text_count: usize,
    text_offset: u64,
    geom_count: usize,
    geom_offset: u64,
}

struct ConrodRenderPass<B: Backend> {
    cache: GlyphCache<'static>,
    image: Image<B>,
    upload: Option<Buffer<B>>,
    vertex: Option<Vertex<B>>,
    pool: XfgDescriptorPool<B>,
    sampler: B::Sampler,
    view: B::ImageView,
    set: Option<B::DescriptorSet>,
}

impl<B> ConrodRenderPass<B>
where
    B: Backend,
{
    fn bindings() -> &'static [DescriptorSetLayoutBinding] {
        &[DescriptorSetLayoutBinding {
            binding: 0,
            ty: DescriptorType::CombinedImageSampler,
            count: 1,
            stage_flags: ShaderStageFlags::FRAGMENT,
            immutable_samplers: false,
        }]
    }
}

impl<B> RenderPassDesc<B> for ConrodRenderPass<B>
where
    B: Backend,
{
    fn name() -> &'static str {
        "ConrodRenderPass"
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
        vec![
            Pipeline {
                layout: 0,
                vertices: vec![(
                    PosColor::VERTEX_FORMAT.attributes.into_owned(),
                    PosColor::VERTEX_FORMAT.stride,
                )],
                colors: vec![ColorBlendDesc(ColorMask::ALL, BlendState::ALPHA)],
                depth_stencil: DepthStencilDesc {
                    depth: DepthTest::On {
                        fun: Comparison::LessEqual,
                        write: true,
                    },
                    depth_bounds: false,
                    stencil: StencilTest::Off,
                },
            },
            Pipeline {
                layout: 0,
                vertices: vec![(
                    PosColorTex::VERTEX_FORMAT.attributes.into_owned(),
                    PosColorTex::VERTEX_FORMAT.stride,
                )],
                colors: vec![ColorBlendDesc(ColorMask::ALL, BlendState::ALPHA)],
                depth_stencil: DepthStencilDesc {
                    depth: DepthTest::On {
                        fun: Comparison::LessEqual,
                        write: true,
                    },
                    depth_bounds: false,
                    stencil: StencilTest::Off,
                },
            },
        ]
    }
}

impl<B, T> RenderPass<B, Factory<B>, Scene<B, T, Ui>> for ConrodRenderPass<B>
where
    B: Backend,
{
    fn load_shader_sets<'a>(
        storage: &'a mut Vec<B::ShaderModule>,
        device: &mut Factory<B>,
        scene: &mut Scene<B, T, Ui>,
    ) -> Vec<GraphicsShaderSet<'a, B>> {
        let offset = storage.len();

        storage.push(
            device
                .create_shader_module(include_bytes!("geom/vert.spv"))
                .unwrap(),
        );
        storage.push(
            device
                .create_shader_module(include_bytes!("geom/frag.spv"))
                .unwrap(),
        );
        storage.push(
            device
                .create_shader_module(include_bytes!("text/vert.spv"))
                .unwrap(),
        );
        storage.push(
            device
                .create_shader_module(include_bytes!("text/frag.spv"))
                .unwrap(),
        );

        vec![
            GraphicsShaderSet {
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
            },
            GraphicsShaderSet {
                vertex: EntryPoint {
                    entry: "main",
                    module: &storage[offset + 2],
                    specialization: &[],
                },
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(EntryPoint {
                    entry: "main",
                    module: &storage[offset + 3],
                    specialization: &[],
                }),
            },
        ]
    }

    fn build<I>(
        _sampled: I,
        _storage: I,
        factory: &mut Factory<B>,
        _scene: &mut Scene<B, T, Ui>,
    ) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<B::ImageView>,
    {
        let mut pool = XfgDescriptorPool::new();

        let sampler = factory.create_sampler(image::SamplerInfo::new(
            image::Filter::Linear,
            image::WrapMode::Clamp,
        ));

        let image = factory
            .create_image(
                image::Kind::D2(640, 480, 1, 1),
                1,
                Format::R8Unorm,
                image::Tiling::Linear,
                image::StorageFlags::empty(),
                image::Usage::TRANSFER_DST | image::Usage::SAMPLED,
                Properties::DEVICE_LOCAL,
            )
            .unwrap();

        let view = factory
            .create_image_view(
                image.borrow(),
                image::ViewKind::D2,
                Format::R8Unorm,
                Swizzle::NO,
                image::SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            )
            .unwrap();

        ConrodRenderPass {
            cache: ::rusttype::gpu_cache::CacheBuilder {
                width: 640,
                height: 480,
                scale_tolerance: 0.1,
                position_tolerance: 0.1,
                pad_glyphs: true,
            }.build(),
            sampler,
            image,
            view,
            pool,
            set: None,
            upload: None,
            vertex: None,
        }
    }

    fn prepare<A, S>(
        &mut self,
        sets: &A,
        cbuf: &mut CommandBuffer<B, Graphics>,
        factory: &mut Factory<B>,
        scene: &Scene<B, T, Ui>,
    ) where
        A: Index<usize>,
        A::Output: Index<usize, Output = S>,
        S: Borrow<B::DescriptorSetLayout>,
    {
        trace!("Prepare UI rendering");

        if scene.other.is_none() {
            return;
        }

        let set = sets[0][0].borrow();

        let ref mut pool = self.pool;
        let ref view = self.view;
        let ref sampler = self.sampler;

        self.set.get_or_insert_with(|| {
            let set = pool.allocate(factory, set, Self::bindings());
            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: &set,
                binding: 0,
                array_offset: 0,
                descriptors: Some(Descriptor::CombinedImageSampler(
                    view,
                    image::Layout::General,
                    sampler,
                )),
            }));
            set
        });

        let mut text_count = 0;
        let mut geom_count = 0;

        let ui = scene.other.as_ref().unwrap();

        let mut primitives = ui.draw();
        while let Some(primitive) = primitives.next() {
            match primitive.kind {
                PrimitiveKind::Text {
                    color,
                    text,
                    font_id,
                } => {
                    let glyphs = text.positioned_glyphs(1.0);
                    for glyph in glyphs {
                        trace!("Enqueue glyph: {:?}", glyph.pixel_bounding_box());
                        self.cache.queue_glyph(0, glyph.clone());
                    }
                    text_count += glyphs.len() * 6;
                }
                PrimitiveKind::Rectangle { color } => {
                    geom_count += 6;
                }
                PrimitiveKind::TrianglesSingleColor { triangles, .. } => {
                    geom_count += triangles.len() * 3;
                }
                PrimitiveKind::TrianglesMultiColor { triangles } => {
                    geom_count += triangles.len() * 3;
                }
                _ => {}
            }
        }

        let ref mut upload = self.upload;
        let ref image = self.image;

        let mut rects = Vec::new();
        let mut bytes = Vec::new();

        // Update glyphs
        self.cache
            .cache_queued(|rect, data| {
                rects.push((rect, data.len() as u32));
                bytes.extend_from_slice(data);
            })
            .unwrap();

        if !rects.is_empty() {
            if let Some(buffer) = upload.take() {
                if buffer.block().size() >= bytes.len() as u64 {
                    *upload = Some(buffer);
                }
            }

            let upload = &*upload.get_or_insert_with(|| {
                factory
                    .create_buffer(
                        bytes.len() as u64,
                        buffer::Usage::TRANSFER_SRC | buffer::Usage::TRANSFER_DST,
                        Properties::DEVICE_LOCAL,
                    )
                    .unwrap()
            });

            cbuf.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::TRANSFER,
                Dependencies::empty(),
                Some(Barrier::Buffer {
                    states: buffer::Access::TRANSFER_READ..buffer::Access::TRANSFER_WRITE,
                    target: upload.borrow(),
                }),
            );

            cbuf.update_buffer(upload.borrow(), 0, &bytes);

            cbuf.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::TRANSFER,
                Dependencies::empty(),
                Some(Barrier::Buffer {
                    states: buffer::Access::TRANSFER_WRITE..buffer::Access::TRANSFER_READ,
                    target: upload.borrow(),
                }),
            );

            // Prepare image for transfer
            cbuf.pipeline_barrier(
                PipelineStage::FRAGMENT_SHADER..PipelineStage::TRANSFER,
                Dependencies::empty(),
                Some(Barrier::Image {
                    states: (image::Access::SHADER_READ, image::Layout::General)
                        ..(image::Access::SHADER_READ, image::Layout::General),
                    target: image.borrow(),
                    range: image::SubresourceRange {
                        aspects: Aspects::COLOR,
                        levels: 0..1,
                        layers: 0..1,
                    },
                }),
            );
            cbuf.copy_buffer_to_image(
                upload.borrow(),
                image.borrow(),
                image::Layout::General,
                rects.into_iter().scan(0u32, |offset, (rect, len)| {
                    *offset += len;

                    assert_eq!(rect.width() * rect.height(), len);

                    let copy = BufferImageCopy {
                        buffer_offset: (*offset - len) as u64,
                        buffer_width: rect.width(),
                        buffer_height: len,
                        image_layers: image::SubresourceLayers {
                            aspects: Aspects::COLOR,
                            level: 0,
                            layers: 0..1,
                        },
                        image_offset: image::Offset {
                            x: rect.min.x as i32,
                            y: rect.min.y as i32,
                            z: 0,
                        },
                        image_extent: image::Extent {
                            width: rect.width(),
                            height: rect.height(),
                            depth: 1,
                        },
                    };
                    trace!("Copy buffer to image: {:#?}", copy);
                    Some(copy)
                }),
            );

            // Prepare image for render
            cbuf.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                Dependencies::empty(),
                Some(Barrier::Image {
                    states: (image::Access::SHADER_READ, image::Layout::General)
                        ..(image::Access::SHADER_READ, image::Layout::General),
                    target: self.image.borrow(),
                    range: image::SubresourceRange {
                        aspects: Aspects::COLOR,
                        levels: 0..1,
                        layers: 0..1,
                    },
                }),
            );
        }

        let text_size = (text_count * size_of::<PosColorTex>()) as u64;
        let geom_size = (geom_count * size_of::<PosColor>()) as u64;

        // Borrow vertex buffer
        let vertex = {
            if let Some(vertex) = self.vertex.take() {
                if vertex.buffer.block().size() >= text_size + geom_size {
                    self.vertex = Some(vertex);
                }
            }
            self.vertex.get_or_insert_with(|| Vertex {
                buffer: factory
                    .create_buffer(
                        text_size + geom_size,
                        buffer::Usage::VERTEX | buffer::Usage::TRANSFER_DST,
                        Properties::DEVICE_LOCAL,
                    )
                    .unwrap(),
                text_offset: geom_size,
                text_count,
                geom_offset: 0,
                geom_count,
            })
        };

        let vertex_buffer = vertex.buffer.borrow();

        // Prepare vertex buffer for transfer
        cbuf.pipeline_barrier(
            PipelineStage::VERTEX_INPUT..PipelineStage::TRANSFER,
            Dependencies::empty(),
            once(Barrier::Buffer {
                states: buffer::Access::VERTEX_BUFFER_READ..buffer::Access::TRANSFER_WRITE,
                target: vertex_buffer,
            }),
        );

        let mut text_offset: u64 = vertex.text_offset;
        let mut text_count: usize = 0;
        let mut text_bucket: [PosColorTex; 3 * 64] = unsafe { zeroed() };

        let mut geom_offset: u64 = vertex.geom_offset;
        let mut geom_count: usize = 0;
        let mut geom_bucket: [PosColor; 3 * 64] = unsafe { zeroed() };
        {
            let mut depth = 0.01;

            let mut text_push =
                |pos: [f32; 3],
                 uv: [f32; 2],
                 color: [f32; 4],
                 cbuf: &mut CommandBuffer<B, Graphics>| {
                    // trace!("Text push :{:?} :{:?} :{:?}", pos, uv, color);

                    if text_count == 3 * 64 {
                        let slice: &[u8] = cast_slice(&text_bucket[..]);
                        cbuf.update_buffer(vertex_buffer, text_offset, slice);
                        text_offset += slice.len() as u64;
                        text_count = 0;
                    }

                    text_bucket[text_count] = PosColorTex {
                        position: Position([pos[0] / 320.0 - 1.0, pos[1] / 240.0 - 1.0, pos[2]]),
                        color: Color(color),
                        tex: TexCoord([uv[0], uv[1]]),
                    };
                    text_count += 1;
                };

            let mut geom_push =
                |pos: [f32; 3], color: [f32; 4], cbuf: &mut CommandBuffer<B, Graphics>| {
                    // trace!("Geom push :{:?} :{:?}", pos, color);

                    if geom_count == 3 * 64 {
                        let slice: &[u8] = cast_slice(&geom_bucket[..]);
                        cbuf.update_buffer(vertex_buffer, geom_offset, slice);
                        geom_offset += slice.len() as u64;
                        geom_count = 0;
                    }

                    geom_bucket[geom_count] = PosColor {
                        position: Position([pos[0] / 320.0, pos[1] / 240.0, pos[2]]),
                        color: Color(color),
                    };

                    geom_count += 1;
                };

            let mut primitives = ui.draw();
            while let Some(primitive) = primitives.next() {
                match primitive.kind {
                    PrimitiveKind::Text {
                        color,
                        text,
                        font_id,
                    } => {
                        let color = color.to_fsa();
                        let glyphs = text.positioned_glyphs(1.0);
                        for glyph in glyphs {
                            if let Some((uv, pos)) = self.cache.rect_for(0, glyph).unwrap() {
                                text_push(
                                    [pos.min.x as f32, pos.min.y as f32, depth],
                                    [uv.min.x, uv.min.y],
                                    color,
                                    cbuf,
                                );
                                text_push(
                                    [pos.min.x as f32, pos.max.y as f32, depth],
                                    [uv.min.x, uv.max.y],
                                    color,
                                    cbuf,
                                );
                                text_push(
                                    [pos.max.x as f32, pos.max.y as f32, depth],
                                    [uv.max.x, uv.max.y],
                                    color,
                                    cbuf,
                                );
                                text_push(
                                    [pos.max.x as f32, pos.max.y as f32, depth],
                                    [uv.max.x, uv.max.y],
                                    color,
                                    cbuf,
                                );
                                text_push(
                                    [pos.max.x as f32, pos.min.y as f32, depth],
                                    [uv.max.x, uv.min.y],
                                    color,
                                    cbuf,
                                );
                                text_push(
                                    [pos.min.x as f32, pos.min.y as f32, depth],
                                    [uv.min.x, uv.min.y],
                                    color,
                                    cbuf,
                                );
                            }
                        }

                        depth -= 0.00001;
                    }
                    PrimitiveKind::Rectangle { color } => {
                        trace!("Render rect {:?}:{:?}", primitive.rect, color);
                        let pos = primitive.rect;
                        let color = color.to_fsa();
                        geom_push([pos.x.start as f32, pos.y.start as f32, depth], color, cbuf);
                        geom_push([pos.x.start as f32, pos.y.end as f32, depth], color, cbuf);
                        geom_push([pos.x.end as f32, pos.y.end as f32, depth], color, cbuf);
                        geom_push([pos.x.end as f32, pos.y.end as f32, depth], color, cbuf);
                        geom_push([pos.x.end as f32, pos.y.start as f32, depth], color, cbuf);
                        geom_push([pos.x.start as f32, pos.y.start as f32, depth], color, cbuf);

                        depth -= 0.00001;
                    }
                    PrimitiveKind::TrianglesSingleColor { color, triangles } => {
                        trace!("Render triangles {:?}:{:?}", triangles, color);

                        let color: [f32; 4] = color.into();

                        for triangle in triangles {
                            for point in &triangle.points() {
                                geom_push([point[0] as f32, point[1] as f32, depth], color, cbuf);
                            }
                        }

                        depth -= 0.00001;
                    }
                    PrimitiveKind::TrianglesMultiColor { triangles } => {
                        trace!("Render triangles {:?}", triangles);

                        for triangle in triangles {
                            for &(point, color) in &triangle.0 {
                                let color: [f32; 4] = color.into();
                                geom_push([point[0] as f32, point[1] as f32, depth], color, cbuf);
                            }
                        }

                        depth -= 0.00001;
                    }
                    PrimitiveKind::Image { .. } => {
                        warn!("Can't render images yet");
                    }
                    PrimitiveKind::Other(other) => {
                        warn!("Can't render custom widgets yet : {:?}", other.type_id);
                    }
                }
            }
        }

        if text_count > 0 {
            cbuf.update_buffer(
                vertex_buffer,
                text_offset,
                cast_slice(&text_bucket[..text_count]),
            );
        }

        if geom_count > 0 {
            cbuf.update_buffer(
                vertex_buffer,
                geom_offset,
                cast_slice(&geom_bucket[..geom_count]),
            );
        }

        cbuf.pipeline_barrier(
            PipelineStage::TRANSFER..PipelineStage::VERTEX_INPUT,
            Dependencies::empty(),
            once(Barrier::Buffer {
                states: buffer::Access::TRANSFER_WRITE..buffer::Access::VERTEX_BUFFER_READ,
                target: vertex_buffer,
            }),
        );
    }

    fn draw<L, P>(
        &mut self,
        layouts: &L,
        pipelines: &P,
        mut encoder: RenderPassInlineEncoder<B, Primary>,
        scene: &Scene<B, T, Ui>,
    ) where
        L: Index<usize>,
        L::Output: Borrow<B::PipelineLayout>,
        P: Index<usize>,
        P::Output: Borrow<B::GraphicsPipeline>,
    {
        if scene.other.is_none() {
            return;
        }

        let geom_pipeline = pipelines[0].borrow();
        let text_pipeline = pipelines[1].borrow();
        let layout = layouts[0].borrow();

        let vertex = self.vertex.as_ref().unwrap();

        if vertex.geom_count > 0 {
            encoder.bind_graphics_pipeline(geom_pipeline);
            encoder.bind_vertex_buffers(
                0,
                VertexBufferSet(vec![(vertex.buffer.borrow(), vertex.geom_offset)]),
            );
            encoder.draw(0..vertex.geom_count as u32, 0..1);
        }

        if vertex.text_count > 0 {
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                Some(self.set.as_ref().unwrap()),
                empty::<DescriptorSetOffset>(),
            );

            encoder.bind_graphics_pipeline(text_pipeline);
            encoder.bind_vertex_buffers(
                0,
                VertexBufferSet(vec![(vertex.buffer.borrow(), vertex.text_offset)]),
            );
            encoder.draw(0..vertex.text_count as u32, 0..1);
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _scene: &mut Scene<B, T, Ui>) {}
}

#[inline(always)]
fn constants<'a>(value: &'a [u8]) -> &'a [u32] {
    use std::slice::from_raw_parts;
    assert_eq!(value.len() % 4, 0);
    unsafe { from_raw_parts(value.as_ptr() as *const u32, value.len() / 4) }
}

fn graph<B>(
    kind: image::Kind,
    surface_format: Format,
    graph: &mut XfgGraphBuilder<B, (), Ui>,
) -> ImageId
where
    B: Backend,
{
    let ui = graph.create_image(
        kind,
        surface_format,
        Some(ClearValue::Color(ClearColor::Float([0.2, 0.3, 0.4, 1.0]))),
    );
    let depth = graph.create_image(
        kind,
        Format::D32Float,
        Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
    );

    graph.add_node(ConrodRenderPass::builder().with_image(ui).with_image(depth));

    ui
}

fn fill<B>(scene: &mut Scene<B, (), Ui>, device: &mut Factory<B>)
where
    B: Backend,
{
    use conrod::{
        color, text::Font, widget::{Button, Canvas, Text}, Colorable, Labelable, Positionable,
        Sizeable, Ui, UiBuilder, Widget,
    };

    use std::fs::File;

    widget_ids!(struct Ids {
        canvas,
        text,
        button,
    });

    let mut oval_range = (0.25, 0.75);

    let mut ui = UiBuilder::new([640.0, 480.0]).build();

    ui.fonts
        .insert_from_file(format!("{}/examples/ui/Orbitron Medium.ttf", env!("CARGO_MANIFEST_DIR")))
        .unwrap();

    let ids = Ids::new(ui.widget_id_generator());

    {
        let ui = &mut ui.set_widgets();

        Canvas::new().pad(40.0).set(ids.canvas, ui);

        Button::new()
            .middle_of(ids.canvas)
            .w_h(100.0, 32.0)
            .color(Rgba(0.2, 0.3, 0.8, 1.0).into())
            .hover_color(Rgba(0.8, 0.3, 0.2, 1.0).into())
            .press_color(Rgba(0.3, 0.8, 0.2, 1.0).into())
            .label("Button")
            .label_color(Rgba(0.0, 0.0, 0.0, 1.0).into())
            .set(ids.button, ui);

        Text::new("QwerTy")
            .top_right_of(ids.canvas)
            .color(Rgba(0.2, 0.3, 0.8, 1.0).into())
            .set(ids.text, ui);
    }

    scene.other = Some(ui);
}

fn main() {
    run(graph, fill);
}
