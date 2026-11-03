#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

layout(set = 2, binding = 0) uniform sampler2D sourceTexture;
layout(set = 3, binding = 0) uniform PresentUniforms {
    vec4 uvRect;
};

void main() {
    color = texture(sourceTexture, uvRect.xy + uv * uvRect.zw);
}
