#include "meshview/texture.hpp"
#include <iostream>
#include <GL/glew.h>
#include "stb_image.h"

namespace meshview {

// *** Texture ***
Texture::Texture(const std::string& path, bool flip, int type)
    : type(type), path(path), fallback_color(/*pink*/ 1.f, 0.75f, 0.8f), flip(flip) {
}

Texture::Texture(const Eigen::Ref<const Vector3f>& color, int type) :
    type(type), fallback_color(color) {
}

Texture::~Texture() {
    if (~id) glDeleteTextures(1, &id);
}

void Texture::load() {
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    bool success = false;
    if (path.size()) {
        stbi_set_flip_vertically_on_load(flip);
        int width, height, nrChannels;
        unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            GLenum format;
            if (nrChannels == 1)
                format = GL_RED;
            else if (nrChannels == 3)
                format = GL_RGB;
            else if (nrChannels == 4)
                format = GL_RGBA;

            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height,
                    0, format, GL_UNSIGNED_BYTE, data);

            glGenerateMipmap(GL_TEXTURE_2D);
            stbi_image_free(data);
            success = true;
        } else {
            std::cerr << "Failed to load texture " << path << ", using fallback color\n";
        }
    }
    if (!success) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1,
                0, GL_RGB, GL_FLOAT, fallback_color.data());
        glGenerateMipmap(GL_TEXTURE_2D);
    }
}

}  // namespace
