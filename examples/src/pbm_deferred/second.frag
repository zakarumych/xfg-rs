#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0, rgba32f) uniform readonly image2D input_albedo_roughness;
layout(set = 0, binding = 1, rgba32f) uniform readonly image2D input_emission_metallic;
layout(set = 0, binding = 2, rgba32f) uniform readonly image2D input_normal_ambient_occlusion;
layout(set = 0, binding = 3, rgba32f) uniform readonly image2D input_position;

layout(set = 0, binding = 4) uniform FragmentArgs {
    vec3 light_position;
    float _pad0;
    vec3 light_color;
    float _pad1;
    vec3 camera_position;
    float _pad2;
    vec3 ambient_light;
    float _pad3;
};

layout(pixel_center_integer) in vec4 gl_FragCoord;
layout(location = 0) out vec4 out_color;

const float PI = 3.14159265359;

float calc_normal_distribution(vec3 N, vec3 H, float a) {
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return (a2 + 0.0000001) / denom;
}

float calc_geometry(float NdotV, float NdotL, float r2) {
    float a1 = r2 + 1.0;
    float k = a1 * a1 / 8.0;
    float denom = NdotV * (1.0 - k) + k;
    float ggx1 = NdotV / denom;
    denom = NdotL * (1.0 - k) + k;
    float ggx2 = NdotL / denom;
    return ggx1 * ggx2;
}

vec3 calc_fresnel(float HdotV, vec3 fresnel_base) {
    return fresnel_base + (1.0 - fresnel_base) * pow(1.0 - HdotV, 5.0);
}

vec4 read_local(readonly image2D image) {
    return imageLoad(image, ivec2(gl_FragCoord.xy));
}

void main() {
    vec4 albedo_roughness = read_local(input_albedo_roughness);
    vec3 albedo = albedo_roughness.rgb;
    float roughness = albedo_roughness.a;

    vec4 emission_metallic = read_local(input_emission_metallic);
    vec3 emission = emission_metallic.rgb;
    float metallic = emission_metallic.a;

    vec4 normal_ambient_occlusion = read_local(input_normal_ambient_occlusion);
    vec3 normal = normal_ambient_occlusion.rgb;
    float ambient_occlusion = normal_ambient_occlusion.a;

    vec4 position_depth = read_local(input_position);
    vec3 position = position_depth.xyz;
    // gl_FragDepth = position_depth.w;

    float roughness2 = roughness * roughness;
    vec3 fresnel_base = mix(vec3(0.04), albedo, metallic);

    vec3 view_direction = normalize(camera_position - position.xyz);
    vec3 light_direction = normalize(light_position.xyz - position.xyz);
    float intensity = 1.0 / dot(light_direction, light_direction);

    vec3 halfway = normalize(view_direction + light_direction);
    float normal_distribution = calc_normal_distribution(normal, halfway, roughness2);

    float NdotV = max(dot(normal, view_direction), 0.0);
    float NdotL = max(dot(normal, light_direction), 0.0);
    float HdotV = max(dot(halfway, view_direction), 0.0);
    float geometry = calc_geometry(NdotV, NdotL, roughness2);

    vec3 fresnel = calc_fresnel(HdotV, fresnel_base);
    vec3 diffuse = vec3(1.0) - fresnel;
    diffuse *= 1.0 - metallic;

    vec3 nominator = normal_distribution * geometry * fresnel;
    float denominator = 4 * NdotV * NdotL + 0.0001;
    vec3 specular = nominator / denominator;

    vec3 lighted = (diffuse * albedo / PI + specular) * light_color.rgb * intensity * NdotL;

    vec3 ambient = ambient_light * albedo * ambient_occlusion;
    vec3 color = ambient + lighted + emission;

    out_color = vec4(color, 1.0);
}
