#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D glyphs;

void main() {
    out_color = color * texture(glyphs, uv);
}
