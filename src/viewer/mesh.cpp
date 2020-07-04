#include "mesh.hpp"
#include <GL/glew.h>

namespace smpl {

Mesh::Mesh(size_t num_verts, size_t num_triangles) : num_verts(num_verts),
    num_triangles(num_triangles) {
}

void Mesh::draw() {
	init_or_update();

	// bind appropriate textures
	unsigned int diffuse_nr  = 1;
	unsigned int specular_nr = 1;
	unsigned int normal_nr   = 1;
	unsigned int height_nr   = 1;
	for(unsigned int i = 0; i < textures.size(); i++)
	{
		glActiveTexture(GL_TEXTURE0 + i); // active proper texture unit before binding
		// retrieve texture number (the N in diffuse_textureN)
		std::string number;
		std::string name = textures[i].type;
		if(name == "texture_diffuse")
			number = std::to_string(diffuse_nr++);
		else if(name == "texture_specular")
			number = std::to_string(specular_nr++); // transfer unsigned int to stream
		else if(name == "texture_normal")
			number = std::to_string(normal_nr++); // transfer unsigned int to stream
		else if(name == "texture_height")
			number = std::to_string(height_nr++); // transfer unsigned int to stream

		// now set the sampler to the correct texture unit
		glUniform1i(glGetUniformLocation(shader.ID, (name + number).c_str()), i);
		// and finally bind the texture
		glBindTexture(GL_TEXTURE_2D, textures[i].id);
	}

	// draw mesh
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, triangles.rows() * triangles.cols(),
			   	   GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	// always good practice to set everything back to defaults once configured.
	glActiveTexture(GL_TEXTURE0);
}

void Mesh::init_or_update() {
    if (verts.rows() == num_verts) return; // Already initialized
    verts.resize(num_verts, 8);
    triangles.resize(num_triangles, 3);

    const size_t BUF_ROW_SZ = verts.cols() * sizeof(verts(0, 0));
    const size_t BUF_SZ = verts.rows() * BUF_SZ;
    const size_t INDEX_SZ = triangles.rows() * triangles.cols() * sizeof(triangles(0, 0));

    // create buffers/arrays
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    // load data into vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // A great thing about structs is that their memory layout is sequential for all its items.
    // The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
    // again translates to 3/2 floats which translates to a byte array.
    glBufferData(GL_ARRAY_BUFFER, BUF_SZ, (GLvoid*) verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.rows() * sizeof(unsigned int),
                                    triangles.data(), GL_STATIC_DRAW);

    // set the vertex attribute pointers
    // vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, BUF_ROW_SZ, (GLvoid*)0);
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, BUF_ROW_SZ, (GLvoid*)3);
    // vertex texture coords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, BUF_ROW_SZ, (GLvoid*)5);
    glBindVertexArray(0);
}

}  // namespace smpl
