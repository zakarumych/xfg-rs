#version 450 core
#extension GL_ARB_separate_shader_objects : enable

struct PointLight {
    vec4 color;
    vec3 position;
    float _pad;
};

struct DirectionalLight {
    vec4 color;
    vec3 direction;
    float _pad;
};

layout(binding = 1, set = 0) uniform FragmentArgs {
    PointLight plight[32];
    vec3 camera_position;
    int point_light_count;
    vec3 ambient_light;
    float ambient_occlusion;
    vec3 albedo;
    float metallic;
    vec3 emission;
    float roughness;
};

layout(location = 0) in VertexData {
    vec4 position;
    vec3 normal;
} vertex;

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

void main() {
    float roughness2 = roughness * roughness;
    vec3 fresnel_base = mix(vec3(0.04), albedo, metallic);

    vec3 lighted = vec3(0.0);
    for (int i = 0; i < point_light_count; i++) {
        vec3 view_direction = normalize(camera_position - vertex.position.xyz);
        vec3 light_direction = normalize(plight[i].position.xyz - vertex.position.xyz);
        float intensity = 1.0 / dot(light_direction, light_direction);

        vec3 halfway = normalize(view_direction + light_direction);
        float normal_distribution = calc_normal_distribution(vertex.normal, halfway, roughness2);

        float NdotV = max(dot(vertex.normal, view_direction), 0.0);
        float NdotL = max(dot(vertex.normal, light_direction), 0.0);
        float HdotV = max(dot(halfway, view_direction), 0.0);
        float geometry = calc_geometry(NdotV, NdotL, roughness2);

        vec3 fresnel = calc_fresnel(HdotV, fresnel_base);
        vec3 diffuse = vec3(1.0) - fresnel;
        diffuse *= 1.0 - metallic;

        vec3 nominator = normal_distribution * geometry * fresnel;
        float denominator = 4 * NdotV * NdotL + 0.0001;
        vec3 specular = nominator / denominator;

        lighted += (diffuse * albedo / PI + specular) * plight[i].color.rgb * intensity * NdotL;
    }

    vec3 ambient = ambient_light * albedo * ambient_occlusion;
    vec3 color = ambient + lighted + emission;
   
    out_color = vec4(color, 1.0);
}