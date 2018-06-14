#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in InterfaceBlock {
    vec4 color;
    vec2 uv;
} block;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D glyphs;

void main() {
    out_color = vec4(block.color.rgb, block.color.a * texture(glyphs, block.uv).r);
}
