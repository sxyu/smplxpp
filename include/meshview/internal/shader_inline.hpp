#pragma once
#ifndef VIEWER_SHADER_INLINE_91D27C05_59C0_4F9F_A6C5_6AA9E2000CDA
#define VIEWER_SHADER_INLINE_91D27C05_59C0_4F9F_A6C5_6AA9E2000CDA

namespace meshview {

static const char* MESH_VERTEX_SHADER = R"SHADER(
#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;

out vec3 FragPos;
out vec2 TexCoord;
out vec3 Normal;

uniform mat4 M;
uniform mat4 MVP;
uniform mat3 NormalMatrix;

void main() {
    TexCoord = aTexCoord;
    FragPos = (M * vec4(aPosition, 1.0f)).xyz;
    Normal = NormalMatrix * aNormal;
    gl_Position = MVP * vec4(aPosition, 1.0f);
}
)SHADER";

static const char* MESH_FRAGMENT_SHADER = R"SHADER(
#version 330 core

// Ouput data
out vec4 FragColor;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    // Not implemented
    // sampler2D normal;
    // sampler2D height;
    float shininess;
};

struct Light {
    vec3 position;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

// Interpolated position (world)
in vec3 FragPos;
// UV coords
in vec2 TexCoord;
// Normal vector (world)
in vec3 Normal;

// Camera position (world)
uniform vec3 viewPos;

// Material info
uniform Material material;
// Light info
uniform Light light;

void main(){
    // vec3(1.0f, 0.5f, 0.31f)
    vec3 objectColor = texture(material.diffuse, TexCoord).rgb;

    // Ambient shading
    vec3 ambient = light.ambient * objectColor;

    // Diffuse shading
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0f);
    vec3 diffuse = light.diffuse * diff * objectColor;

    // Specular shading
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = light.specular * spec * texture(material.specular, TexCoord).rgb;

    // Finish
    FragColor = vec4(ambient + diffuse + specular, 1.0f);
}
)SHADER";

static const char* POINTCLOUD_VERTEX_SHADER = R"SHADER(
#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aColor;

out vec3 Color;

uniform mat4 M;
uniform mat4 MVP;
void main() {
    Color = aColor;
    gl_Position = MVP * vec4(aPosition, 1.0f);
}
)SHADER";

static const char* POINTCLOUD_FRAGMENT_SHADER = R"SHADER(
#version 330 core

out vec4 FragColor; // Ouput data
in vec3 Color; // Color
void main(){
    // Finish
    FragColor = vec4(Color, 1.0f);
}
)SHADER";

}  // namespace meshview

#endif  // ifndef VIEWER_SHADER_INLINE_91D27C05_59C0_4F9F_A6C5_6AA9E2000CDA
