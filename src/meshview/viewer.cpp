#include "meshview/viewer.hpp"

#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "meshview/util.hpp"
#include "meshview/internal/shader_inline.hpp"

namespace {
void error_callback(int error, const char *description) {
    std::cerr << description << "\n";
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    bool prevent_default = false;

    meshview::Viewer& viewer = *reinterpret_cast<meshview::Viewer*>(
            glfwGetWindowUserPointer(window));
    for (auto& cb : viewer.key_callbacks) {
        prevent_default |= !cb(key, action, mods);
    }

    if (prevent_default) return;
    if (action == GLFW_PRESS) {
        switch(key) {
            case GLFW_KEY_ESCAPE: case 'Q':
                glfwSetWindowShouldClose(window, GL_TRUE); break;
            case 'Z': viewer.camera.reset_view(); break;
            case 'W':
                      viewer.wireframe = !viewer.wireframe;
                      glPolygonMode(GL_FRONT_AND_BACK, viewer.wireframe ? GL_LINE : GL_FILL);
                      break;
            case 'C':
                      viewer.cull_face = !viewer.cull_face;
                      if (viewer.cull_face) glEnable(GL_CULL_FACE);
                      else glDisable(GL_CULL_FACE);
                      break;
            case 'M':
                      if (glfwGetWindowAttrib(window, GLFW_MAXIMIZED) == GLFW_TRUE) {
                          glfwRestoreWindow(window);
                      } else {
                          glfwMaximizeWindow(window);
                      }
                      break;
            case 'F':
                      {
                          int* backup = viewer._fullscreen_backup;
                          if (viewer._fullscreen) {
                              glfwSetWindowMonitor(window, nullptr,
                                      backup[0], backup[1], backup[2], backup[3], 0);
                              viewer._fullscreen = false;
                          } else {
                              glfwGetWindowPos(window, &backup[0],
                                      &backup[1] );
                              glfwGetWindowSize(window, &backup[2],
                                      &backup[3] );
                              const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
                              glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(),
                                      0, 0, mode->width, mode->height, 0);
                              viewer._fullscreen = true;
                          }
                      }
                      break;
            case 'H':
                  std::cout <<
R"HELP(Meshview help (c) Alex Yu 2020
left click + drag:         rotate view
shift + left click + drag: pan view
middle click + drag:       pan view (alt)
Z:                         reset view
W:                         toggle wireframe
C:                         toggle backface culling
M:                         toggle maximize window (may not work on some systems)
F:                         toggle fullscreen window
)HELP";
                      break;
        }
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    bool prevent_default = false;

    meshview::Viewer& viewer = *reinterpret_cast<meshview::Viewer*>(
            glfwGetWindowUserPointer(window));
    glfwGetCursorPos(window, &viewer._mouse_x, &viewer._mouse_y);
    for (auto& cb : viewer.mouse_button_callbacks) {
        prevent_default |= !cb(button, action, mods);
    }
    if (action == GLFW_RELEASE) {
        viewer._mouse_button = -1;
    }
    if (prevent_default) return;
    if (action == GLFW_PRESS) {
        viewer._mouse_button = button;
        viewer._mouse_mods = mods;
    }
}

void mouse_move_callback(GLFWwindow* window, double x, double y) {
    bool prevent_default = false;

    meshview::Viewer& viewer = *reinterpret_cast<meshview::Viewer*>(
            glfwGetWindowUserPointer(window));
    double prex = viewer._mouse_x, prey = viewer._mouse_y;
    viewer._mouse_x = x, viewer._mouse_y = y;
    for (auto& cb : viewer.mouse_move_callbacks) {
        prevent_default |= !cb();
    }
    if (prevent_default) return;
    if (viewer._mouse_button != -1) {
        if ((viewer._mouse_button == GLFW_MOUSE_BUTTON_LEFT &&
            (viewer._mouse_mods & GLFW_MOD_SHIFT)) ||
             viewer._mouse_button == GLFW_MOUSE_BUTTON_MIDDLE) {
            // Pan
            viewer.camera.pan_with_mouse((float)(x - prex), (float)(y - prey));
        } else if (viewer._mouse_button == GLFW_MOUSE_BUTTON_LEFT) {
            viewer.camera.rotate_with_mouse((float)(x - prex), (float)(y - prey));
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    bool prevent_default = false;

    meshview::Viewer& viewer = *reinterpret_cast<meshview::Viewer*>(
            glfwGetWindowUserPointer(window));
    for (auto& cb : viewer.scroll_callbacks) {
        prevent_default |= !cb(xoffset, yoffset);
    }
    if (prevent_default) return;
    viewer.camera.zoom_with_mouse((float)yoffset);
}

// Window resize
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    meshview::Viewer& viewer = *reinterpret_cast<meshview::Viewer*>(
            glfwGetWindowUserPointer(window));
    viewer.camera.set_aspect((float)width / (float)height);
    glViewport(0, 0, width, height);
}
}  // namespace

namespace meshview {

Viewer::Viewer() : _fullscreen(false) {
    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        std::cerr << "GLFW failed to initialize\n";
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    background.setZero();
}

Viewer::~Viewer() {
    glfwTerminate();
}

void Viewer::show() {
    GLFWwindow* window = glfwCreateWindow(_width, _height, "meshview", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "GLFW window creation failed\n";
        return;
    }

    camera.set_aspect((float)_width / (float)_height);

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed\n";
        getchar();
        glfwTerminate();
        return;
    }

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    if (cull_face) glEnable(GL_CULL_FACE);

    // Make shader
    Shader shader; //(util::find_data_file("shaders/meshview.vert"),
    //                util::find_data_file("shaders/meshview.frag");
    // Use inlined version
    shader.compile(DATA_VERTEX_SHADER, DATA_FRAGMENT_SHADER);

    // Events
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, mouse_move_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetWindowUserPointer(window, this);

    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);

    glfwSetWindowTitle(window, title.c_str());

    scene.reset();
    shader.use();
    while (!glfwWindowShouldClose(window)) {
        glClearColor(background[0], background[1], background[2], 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        scene.draw(shader, camera);
        shader.use();

        for (auto& cb: loop_callbacks) {
            cb();
        }

        glfwSwapBuffers(window);
        glfwWaitEvents();
    }
    glfwDestroyWindow(window);
}



}  // namespace meshview
