#pragma once
#ifndef VIEWER_VIEWER_A821B138_6EE4_4B99_9B92_C87508B97197
#define VIEWER_VIEWER_A821B138_6EE4_4B99_9B92_C87508B97197

#include <functional>
#include <vector>

#include "meshview/mesh.hpp"
#include "meshview/camera.hpp"

namespace meshview {

class Viewer {
public:
    Viewer();
    ~Viewer();

    // Show window and start render loop; blocks execution
    // press q/ESC to close window and exit loop
    void show();

    // Shorthand for adding mesh (to Viewer::meshes)
    Mesh& add(Mesh&& mesh);
    Mesh& add(const Mesh& mesh);

    // * The meshes
    std::vector<Mesh> meshes;

    // * Lighting
    // Ambient light color, default 0.2 0.2 0.2
    Vector3f ambient_light_color;

    // Point light position (in VIEW space, so that light follows camera)
    Vector3f light_pos;
    // Light color diffuse/specular, default white
    Vector3f light_color_diffuse;
    Vector3f light_color_specular;

    // * Camera
    Camera camera;

    // * Render params
    // Wireframe mode? (w)
    bool wireframe = false;
    // Backface culling? (c)
    bool cull_face = true;

    // * Aesthetics
    // Window title, updated on show() calls only (i.e. please set before show())
    std::string title = "meshview";

    // Background color
    Vector3f background;

    // * Event callbacks
    // Called after GL cnotext init
    std::function<void()> on_open;
    // Called when window is about to close
    std::function<void()> on_close;
    // Called per iter of render loop
    std::function<void()> on_loop;
    // Called per iter of render loop
    // - if MESHVIEW_IMGUI defined, calls with Dear ImGui frame already set up
    //   ie. ready to use ImGui::Begin etc
    // - else, called after loop_callback
    std::function<void()> on_gui;
    // Called on key up/down/repeat: args (key, action, mods), return false to prevent default
    // see https://www.glfw.org/docs/3.3/group__input.html for info on action
    // see https://www.glfw.org/docs/3.3/group__mods.html on mods
    std::function<bool(int, int, int)> on_key;
    // Called on mouse up/down/repeat: args (button, action, mods), return false to prevent default
    // see https://www.glfw.org/docs/3.3/group__input.html for info on button
    // see https://www.glfw.org/docs/3.3/group__mods.html on mods
    std::function<bool(int, int, int)> on_mouse_button;
    // Called on mouse move: args(x, y) return false to prevent default
    std::function<bool(double, double)> on_mouse_move;
    // Called on mouse scroll: args(xoffset, yoffset) return false to prevent default
    std::function<bool(double, double)> on_scroll;

    // * Dynamic data (advanced, for use in callbacks)
    // Window width/height, as set in system
    // (don't modify after show() and before window close)
    int _width = 1000, _height = 600;

    // Mouse position, if available (don't modify)
    double _mouse_x, _mouse_y;

    // Mouse button and modifier keys; mouse_button is -1 if nothing down (don't modify)
    int _mouse_button = -1, _mouse_mods;

    // Is window in fullscreen? (don't modify)
    bool _fullscreen;

    // Window pos/size prior to full screen
    int _fullscreen_backup[4];

    // Pointer to GLFW window object
    void* _window = nullptr;
};

}  // namespace meshview
#endif  // ifndef VIEWER_VIEWER_A821B138_6EE4_4B99_9B92_C87508B97197
