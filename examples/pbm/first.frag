#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in VertexData {
    vec3 position;
    vec3 normal;
} vertex;

layout(binding = 1, set = 0) uniform FragmentArgs {
    vec3 albedo;
    float roughness;
    vec3 emission;
    float metallic;
    float ambient_occlusion;
};

layout(location = 0) out vec4 albedo_roughness;
layout(location = 1) out vec4 emission_metallic;
layout(location = 2) out vec4 normal_ambient_occlusion;
layout(location = 3) out vec4 position;

void main() {
    albedo_roughness = vec4(albedo, roughness);
    emission_metallic = vec4(emission, metallic);
    normal_ambient_occlusion = vec4(vertex.normal, ambient_occlusion);
    position = vec4(vertex.position, 0.0);
}
