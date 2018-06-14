#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 uv;

layout(location = 0) out InterfaceBlock {
    vec4 color;
    vec2 uv;
} block;

void main() {
    block.uv = uv;
    block.color = color;
    gl_Position = vec4(position, 1.0);
}
