#pragma once
#ifndef VIEWER_SHADER_9845A71E_0422_44A7_9AF9_FAC46ECE9C40
#define VIEWER_SHADER_9845A71E_0422_44A7_9AF9_FAC46ECE9C40

#include <string>
#include <cstdint>

#include "common.hpp"

namespace meshview {

class Shader {
public:
    Shader();

    // Load shader on construction from code
    Shader(const std::string& vertex_code,
           const std::string& fragment_code,
           const std::string& geometry_code = "");

    // Generates the shader on the fly from file
    void load(const std::string& vertex_path,
           const std::string& fragment_path,
           const std::string& geometry_path = "");
    // Generates the shader on the fly from code
    void compile(const std::string& vertex_code,
                 const std::string& fragment_code,
                 const std::string& geometry_code = "");

    // Activate the shader
    void use();

    // Utility uniform functions
    void set_bool(const std::string &name, bool value) const;
    void set_int(const std::string &name, int value) const;
    void set_float(const std::string &name, float value) const;
    void set_vec2(const std::string &name, float x, float y) const;
    void set_vec3(const std::string &name, float x, float y, float z) const;
    void set_vec4(const std::string &name, float x, float y, float z, float w);

    // Eigen helpers
    void set_vec2(const std::string &name, const Eigen::Ref<const Vector2f>& value) const;
    void set_vec3(const std::string &name, const Eigen::Ref<const Vector3f>& value) const;
    void set_vec4(const std::string &name, const Eigen::Ref<const Vector4f>& value) const;
    void set_mat2(const std::string &name, const Eigen::Ref<const Matrix2f> &mat) const;
    void set_mat3(const std::string &name, const Eigen::Ref<const Matrix3f> &mat) const;
    void set_mat4(const std::string &name, const Eigen::Ref<const Matrix4f> &mat) const;

    // GL shader id
    Index id;
};

}  // namespace meshview

#endif  // ifndef VIEWER_SHADER_9845A71E_0422_44A7_9AF9_FAC46ECE9C40
