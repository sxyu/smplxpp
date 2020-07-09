#pragma once
#ifndef MESHVIEW_TEXTURE_4C9E53AB_10FA_4057_8EDB_1E117271A1C3
#define MESHVIEW_TEXTURE_4C9E53AB_10FA_4057_8EDB_1E117271A1C3

#include "common.hpp"
#include <string>

namespace meshview {

// Represents a texture/material
struct Texture {
    // Texture types
    enum {
        TYPE_DIFFUSE,
        TYPE_SPECULAR,
        // TYPE_NORMAL,
        // TYPE_HEIGHT,
        __TYPE_COUNT
    };
    // Texture type names (TYPE_DIFFUSE -> "diffuse" etc)
    static constexpr inline const char* type_to_name(int type) {
        switch(type) {
            case TYPE_SPECULAR: return "specular";
            // case TYPE_NORMAL: return "normal";
            // case TYPE_HEIGHT: return "height";
            default: return "diffuse";
        }
    }

    // Texture from path
    Texture(const std::string& path, bool flip = true, int type = TYPE_DIFFUSE);
    // Texture from solid color (1x1)
    Texture(const Eigen::Ref<const Vector3f>& color, int type = TYPE_DIFFUSE);

    ~Texture();

    // Load texture; need to be called before first use in each context
    // (called by Mesh::reset())
    void load();

    // GL texture id; -1 if unavailable
    Index id = -1;

    // File path (optional)
    std::string path;

    // Color to use if path empty OR failed to load the texture image
    Vector3f fallback_color;

    // Texture type
    int type;

    // Vertical flip on load?
    bool flip;
};

}  // namespace meshview

#endif  // ifndef MESHVIEW_TEXTURE_4C9E53AB_10FA_4057_8EDB_1E117271A1C3
