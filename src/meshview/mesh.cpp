#include "meshview/mesh.hpp"

#include <iostream>
#include <GL/glew.h>
#include "stb_image.h"
#include <Eigen/Geometry>

#include "meshview/util.hpp"

namespace meshview {

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

Mesh::Mesh(size_t num_verts, size_t num_triangles) : num_verts(num_verts),
    num_triangles(num_triangles), VAO(-1) {
    verts.resize(num_verts, verts.ColsAtCompileTime);
    if (~num_triangles) {
        faces.resize(num_triangles, faces.ColsAtCompileTime);
    }
    transform.setIdentity();
}

Mesh::~Mesh() {
    if (~VAO) glDeleteVertexArrays(1, &VAO);
    if (~VBO) glDeleteBuffers(1, &VBO);
    if (~num_triangles && ~EBO) glDeleteBuffers(1, &EBO);
    if (~blank_tex_id) glDeleteTextures(1, &blank_tex_id);
}

void Mesh::draw(const Shader& shader, const Camera& camera) {
    if (!enabled) return;
    init_or_update();

    // Bind appropriate textures
    MeshIndex tex_id = 1;
    bool use_blank_tex = false;;
    for(int ttype = 0; ttype < Texture::__TYPE_COUNT; ++ttype) {
        const char* ttype_name = Texture::type_to_name(ttype);
        auto& tex_vec = textures[ttype];
        MeshIndex cnt  = 0;
        for(size_t i = 0; i < tex_vec.size(); i++, tex_id++) {
            glActiveTexture(GL_TEXTURE0 + tex_id); // Active proper texture unit before binding
            // Retrieve texture number (the N in diffuse_textureN)
            MeshIndex number = 0;

            // Now set the sampler to the correct texture unit
            shader.set_int("material." + std::string(ttype_name)
                    + (cnt ? std::to_string(cnt) : ""), tex_id);
            // And finally bind the texture
            glBindTexture(GL_TEXTURE_2D, tex_vec[i].id);
        }
        if (tex_vec.empty()) {
            gen_blank_texture();
            shader.set_int("material." + std::string(ttype_name), 0);
            use_blank_tex = true;
        }
    }
    if (use_blank_tex) {
        glActiveTexture(GL_TEXTURE0); // Active proper texture unit before binding
        glBindTexture(GL_TEXTURE_2D, blank_tex_id);
    }
    shader.set_float("material.shininess", shininess);

    // Set space transform matrices
    shader.set_mat4("M", transform);
    shader.set_mat4("MVP", camera.proj * camera.view * transform);

    auto normal_matrix = transform.topLeftCorner<3, 3>().inverse().transpose();
    shader.set_mat3("NormalMatrix", normal_matrix);

    // Draw mesh
    glBindVertexArray(VAO);
    if (~num_triangles) {
        glDrawElements(GL_TRIANGLES, faces.size(), GL_UNSIGNED_INT, 0);
    } else {
        glDrawArrays(GL_TRIANGLES, 0, verts.rows());
    }
    glBindVertexArray(0);

    // Always good practice to set everything back to defaults once configured.
    glActiveTexture(GL_TEXTURE0);
}

Mesh& Mesh::estimate_normals() {
    util::estimate_normals(verts_pos(), faces, verts_norm());
    return *this;
}

Mesh& Mesh::set_shininess(float val) {
    shininess = val;
    return *this;
}

Mesh& Mesh::translate(const Eigen::Ref<const Vector3f>& vec) {
    transform.topRightCorner<3, 1>() += vec;
    return *this;
}

Mesh& Mesh::rotate(const Eigen::Ref<const Matrix3f>& mat) {
    transform.topRightCorner<3, 3>() = mat * transform.topRightCorner<3, 3>();
    return *this;
}

Mesh& Mesh::scale(const Eigen::Ref<const Vector3f>& vec) {
    transform.topRightCorner<3, 3>().array().colwise() *= vec.array();
    return *this;
}

Mesh& Mesh::set_transform(const Eigen::Ref<const Matrix4f>& mat) {
    transform = mat;
    return *this;
}

Mesh& Mesh::scale(float val) {
    transform.topRightCorner<3, 3>().array() *= val;
    return *this;
}

void Mesh::reinit() {
    VAO = -1;
    for (auto& tex_vec : textures) {
        for (auto& tex : tex_vec) tex.load();
    }
    blank_tex_id = -1;
}

void Mesh::init_or_update() {
    static const size_t SCALAR_SZ = sizeof(Scalar);
    static const size_t POS_OFFSET = 0;
    static const size_t UV_OFFSET = 3 * SCALAR_SZ;
    static const size_t NORMALS_OFFSET = 5 * SCALAR_SZ;
    static const size_t VERT_INDICES = verts.ColsAtCompileTime;
    static const size_t VERT_SZ = VERT_INDICES * SCALAR_SZ;

    if (~VAO) return; // Already initialized

    if (verts.size() != num_verts * VERT_INDICES) {
        std::cerr << "Invalid vertex buf size, expect " << num_verts * VERT_INDICES << "\n";
        return;
    }
    if (~num_triangles && faces.size() != num_triangles * faces.ColsAtCompileTime) {
        std::cerr << "Invalid indices size, expect " << num_triangles * faces.ColsAtCompileTime
            << "\n";
        return;
    }

    const size_t BUF_SZ = verts.size() * SCALAR_SZ;
    const size_t INDEX_SZ = faces.size() * SCALAR_SZ;

    // create buffers/arrays
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    if (~num_triangles) glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    // load data into vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // A great thing about structs is that their memory layout is sequential for all its items.
    // The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
    // again translates to 3/2 floats which translates to a byte array.
    glBufferData(GL_ARRAY_BUFFER, BUF_SZ, (GLvoid*) verts.data(), GL_STATIC_DRAW);

    if (~num_triangles) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, INDEX_SZ, faces.data(), GL_STATIC_DRAW);
    }

    // set the vertex attribute pointers
    // vertex positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERT_SZ, (GLvoid*)POS_OFFSET);
    // vertex texture coords
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, VERT_SZ, (GLvoid*)UV_OFFSET);
    // vertex normals
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, VERT_SZ, (GLvoid*)NORMALS_OFFSET);
    glBindVertexArray(0);
}

void Mesh::gen_blank_texture() {
    if (~blank_tex_id) return;
    glGenTextures(1, &blank_tex_id);
    glBindTexture(GL_TEXTURE_2D, blank_tex_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    Vector3f white(1.f, 1.f, 1.f);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1,
                0, GL_RGB, GL_FLOAT, white.data());
        glGenerateMipmap(GL_TEXTURE_2D);
}

Mesh Mesh::Triangle(const Eigen::Ref<const Vector3f>& a,
                    const Eigen::Ref<const Vector3f>& b,
                    const Eigen::Ref<const Vector3f>& c) {
    Vector3f n = (b - a).cross(c - b);
    Mesh m(3);
    m.verts <<
        a[0], a[1], a[2], 0.f, 0.f, n[0], n[1], n[2],
        b[0], b[1], b[2], 0.f, 1.f, n[0], n[1], n[2],
        c[0], c[1], c[2], 1.f, 1.f, n[0], n[1], n[2];
    return m;
}

Mesh Mesh::Square(float side_len) {
    Mesh m(4, 2);
    m.faces << 0, 3, 1,
                 1, 3, 2;
    m.verts <<
        side_len,  side_len, 0.f,   1.0f, 1.0f,        0.0f, 0.0f, 1.0f,
        side_len, -side_len, 0.f,   1.0f, 0.0f,        0.0f, 0.0f, 1.0f,
        -side_len, -side_len, 0.f,   0.0f, 0.0f,       0.0f, 0.0f, 1.0f,
        -side_len,  side_len, 0.f,   0.0f, 1.0f,       0.0f, 0.0f, 1.0f;
    return m;
}

Mesh Mesh::Cube(float side_len) {
    Mesh m(36);
    m.verts <<
        // positions                        // uv coords    // normals
        // back
        -side_len, -side_len, -side_len,    0.0f,  0.0f,    0.0f,  0.0f, -1.0f,
         side_len,  side_len, -side_len,    1.0f,  1.0f,    0.0f,  0.0f, -1.0f,
         side_len, -side_len, -side_len,    1.0f,  0.0f,    0.0f,  0.0f, -1.0f,
         side_len,  side_len, -side_len,    1.0f,  1.0f,    0.0f,  0.0f, -1.0f,
        -side_len, -side_len, -side_len,    0.0f,  0.0f,    0.0f,  0.0f, -1.0f,
        -side_len,  side_len, -side_len,    0.0f,  1.0f,    0.0f,  0.0f, -1.0f,

        // front
        -side_len, -side_len,  side_len,    0.0f,  0.0f,    0.0f,  0.0f,  1.0f,
         side_len, -side_len,  side_len,    1.0f,  0.0f,    0.0f,  0.0f,  1.0f,
         side_len,  side_len,  side_len,    1.0f,  1.0f,    0.0f,  0.0f,  1.0f,
         side_len,  side_len,  side_len,    1.0f,  1.0f,    0.0f,  0.0f,  1.0f,
        -side_len,  side_len,  side_len,    0.0f,  1.0f,    0.0f,  0.0f,  1.0f,
        -side_len, -side_len,  side_len,    0.0f,  0.0f,    0.0f,  0.0f,  1.0f,

        // left
        -side_len,  side_len,  side_len,    1.0f,  0.0f,   -1.0f,  0.0f,  0.0f,
        -side_len,  side_len, -side_len,    1.0f,  1.0f,   -1.0f,  0.0f,  0.0f,
        -side_len, -side_len, -side_len,    0.0f,  1.0f,   -1.0f,  0.0f,  0.0f,
        -side_len, -side_len, -side_len,    0.0f,  1.0f,   -1.0f,  0.0f,  0.0f,
        -side_len, -side_len,  side_len,    0.0f,  0.0f,   -1.0f,  0.0f,  0.0f,
        -side_len,  side_len,  side_len,    1.0f,  0.0f,   -1.0f,  0.0f,  0.0f,

         // right
         side_len,  side_len,  side_len,    1.0f,  0.0f,    1.0f,  0.0f,  0.0f,
         side_len, -side_len, -side_len,    0.0f,  1.0f,    1.0f,  0.0f,  0.0f,
         side_len,  side_len, -side_len,    1.0f,  1.0f,    1.0f,  0.0f,  0.0f,
         side_len, -side_len, -side_len,    0.0f,  1.0f,    1.0f,  0.0f,  0.0f,
         side_len,  side_len,  side_len,    1.0f,  0.0f,    1.0f,  0.0f,  0.0f,
         side_len, -side_len,  side_len,    0.0f,  0.0f,    1.0f,  0.0f,  0.0f,

         // bottom
        -side_len, -side_len, -side_len,    0.0f,  1.0f,    0.0f, -1.0f,  0.0f,
         side_len, -side_len, -side_len,    1.0f,  1.0f,    0.0f, -1.0f,  0.0f,
         side_len, -side_len,  side_len,    1.0f,  0.0f,    0.0f, -1.0f,  0.0f,
         side_len, -side_len,  side_len,    1.0f,  0.0f,    0.0f, -1.0f,  0.0f,
        -side_len, -side_len,  side_len,    0.0f,  0.0f,    0.0f, -1.0f,  0.0f,
        -side_len, -side_len, -side_len,    0.0f,  1.0f,    0.0f, -1.0f,  0.0f,

        // top
        -side_len,  side_len, -side_len,    0.0f,  1.0f,    0.0f,  1.0f,  0.0f,
         side_len,  side_len,  side_len,    1.0f,  0.0f,    0.0f,  1.0f,  0.0f,
         side_len,  side_len, -side_len,    1.0f,  1.0f,    0.0f,  1.0f,  0.0f,
         side_len,  side_len,  side_len,    1.0f,  0.0f,    0.0f,  1.0f,  0.0f,
        -side_len,  side_len, -side_len,    0.0f,  1.0f,    0.0f,  1.0f,  0.0f,
        -side_len,  side_len,  side_len,    0.0f,  0.0f,    0.0f,  1.0f,  0.0f;
    return m;
}

}  // namespace meshview
