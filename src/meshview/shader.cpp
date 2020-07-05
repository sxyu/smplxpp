// Copied & adapted from learnopengl.com
#include "meshview/shader.hpp"

#include <GL/glew.h>
#include <fstream>
#include <iostream>
#include <sstream>

namespace meshview {

namespace {

void check_compile_errors(GLuint shader, const std::string& type) {
    GLint success;
    GLchar infoLog[1024];
    if(type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if(!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type <<
                "\n" << infoLog <<
                "\n -- ---------------------------------------------------  " << std::endl;
        }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if(!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type <<
                "\n" << infoLog <<
                "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}

}  // namespace

Shader::Shader() : id((MeshIndex)-1) {}

Shader::Shader(const std::string& vertex_path,
        const std::string& fragment_path,
        const std::string& geometry_path) {
    load(vertex_path, fragment_path, geometry_path);
}

void Shader::load(const std::string& vertex_path,
        const std::string& fragment_path,
        const std::string& geometry_path) {
    // Retrieve the vertex/fragment source code from file_path
    std::string vertex_code, fragment_code, geometry_code;
    try {
        // open files
        std::ifstream v_shader_file(vertex_path);
        std::ifstream f_shader_file(fragment_path);
        std::stringstream v_shader_stream, f_shader_stream;
        // read file's buffer contents into streams
        v_shader_stream << v_shader_file.rdbuf();
        f_shader_stream << f_shader_file.rdbuf();
        // close file handlers
        v_shader_file.close();
        f_shader_file.close();
        // convert stream into string
        vertex_code = v_shader_stream.str();
        fragment_code = f_shader_stream.str();
        // if geometry shader path is present, also load a geometry shader
        if(geometry_path.size()) {
            std::ifstream g_shader_file(geometry_path);
            std::stringstream g_shader_stream;
            g_shader_stream << g_shader_file.rdbuf();
            g_shader_file.close();
            geometry_code = g_shader_stream.str();
        }
    } catch (std::ifstream::failure& e) {
        std::cout << "ERROR: shader file not succesfully read, "
                     "did you delete data/shaders/?" << std::endl;
        return;
    }
    compile(vertex_code, fragment_code, geometry_code);
}

void Shader::compile(const std::string& vertex_code,
        const std::string& fragment_code,
        const std::string& geometry_code) {
    const char* v_shader_code = vertex_code.c_str();
    const char * f_shader_code = fragment_code.c_str();
    // 2. compile shaders
    MeshIndex vertex, fragment;
    // vertex shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &v_shader_code, NULL);
    glCompileShader(vertex);
    check_compile_errors(vertex, "VERTEX");
    // fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &f_shader_code, NULL);
    glCompileShader(fragment);
    check_compile_errors(fragment, "FRAGMENT");
    // if geometry shader is given, compile geometry shader
    MeshIndex geometry;
    if(geometry_code.size()) {
        const char * g_shader_code = geometry_code.c_str();
        geometry = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometry, 1, &g_shader_code, NULL);
        glCompileShader(geometry);
        check_compile_errors(geometry, "GEOMETRY");
    }
    // shader program
    id = glCreateProgram();
    glAttachShader(id, vertex);
    glAttachShader(id, fragment);
    if(geometry_code.size())
        glAttachShader(id, geometry);
    glLinkProgram(id);
    check_compile_errors(id, "PROGRAM");
    // delete the shaders as they're linked into our program now and no longer necessery
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    if(geometry_code.size()) glDeleteShader(geometry);
}

void Shader::use() {
    if (id == (MeshIndex)-1) {
        std::cerr << "Shader is not initialized\n";
        return;
    }
    glUseProgram(id);
}

void Shader::set_bool(const std::string &name, bool value) const {
    glUniform1i(glGetUniformLocation(id, name.c_str()), (int)value);
}

void Shader::set_int(const std::string &name, int value) const {
    glUniform1i(glGetUniformLocation(id, name.c_str()), value);
}
void Shader::set_float(const std::string &name, float value) const {
    glUniform1f(glGetUniformLocation(id, name.c_str()), value);
}
void Shader::set_vec2(const std::string &name, float x, float y) const {
    glUniform2f(glGetUniformLocation(id, name.c_str()), x, y);
}
void Shader::set_vec3(const std::string &name, float x, float y, float z) const {
    glUniform3f(glGetUniformLocation(id, name.c_str()), x, y, z);
}
void Shader::set_vec4(const std::string &name, float x, float y, float z, float w) {
    glUniform4f(glGetUniformLocation(id, name.c_str()), x, y, z, w);
}

void Shader::set_vec2(const std::string &name, const Eigen::Ref<const Vector2f> &value) const {
    glUniform2fv(glGetUniformLocation(id, name.c_str()), 1, value.data());
}
void Shader::set_vec3(const std::string &name, const Eigen::Ref<const Vector3f> &value) const {
    glUniform3fv(glGetUniformLocation(id, name.c_str()), 1, value.data());
}
void Shader::set_vec4(const std::string &name, const Eigen::Ref<const Vector4f> &value) const {
    glUniform4fv(glGetUniformLocation(id, name.c_str()), 1, value.data());
}
void Shader::set_mat2(const std::string &name, const Eigen::Ref<const Matrix2f> &mat) const {
    glUniformMatrix2fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, mat.data());
}
void Shader::set_mat3(const std::string &name, const Eigen::Ref<const Matrix3f> &mat) const {
    glUniformMatrix3fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, mat.data());
}
void Shader::set_mat4(const std::string &name, const Eigen::Ref<const Matrix4f> &mat) const {
    glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, mat.data());
}
}  // namespace meshview

#include <fstream>
#include <sstream>
#include <iostream>
