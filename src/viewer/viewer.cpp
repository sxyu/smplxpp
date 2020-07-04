#include "viewer.hpp"

#include <cstdio>

#include <GL/glew.h>

#include <GLFW/glfw3.h>
// #include <glm/glm.hpp>

namespace {
void error_callback(int error, const char *description) {
    fputs(description, stderr);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

void render(GLFWwindow* window) {
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_ACCUM_BUFFER_BIT);
}
}  // namespace

namespace smpl {

Viewer::Viewer() {
    GLFWwindow *window;

    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    window = glfwCreateWindow(1000, 600, "SMPL++X Viewer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        fputs("GLFW failed to initialize\n", stderr);
        return;
    }

    glewExperimental = GL_TRUE;
    glewInit();

    glfwMakeContextCurrent(window);
    p_window = reinterpret_cast<void*>(window);
    fprintf(stderr, "SMPL++X Viewer: Using OpenGL version %s\n", glGetString(GL_VERSION));
}

Viewer::~Viewer() {
    glfwDestroyWindow(reinterpret_cast<GLFWwindow*>(p_window));
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

void Viewer::spin() {
    GLFWwindow* window = reinterpret_cast<GLFWwindow*>(p_window);
    while (!glfwWindowShouldClose(window)) {
        render(window);

        glfwSwapBuffers(window);
        glfwPollEvents();
        //glfwWaitEvents();
    }
}

}  // namespace smpl
