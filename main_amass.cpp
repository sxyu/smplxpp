// Displays AMASS sequence in a OpenGL 3D viewer
// 2 optional arguments:
// 1. model type, default H
//    options: S H X (SMPL SMPL-H SMPL-X)
// 2. sequence (AMASS .npz) path. If not specified,
//    opens a blank viewer with option to browse and load a npz
#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "imfilebrowser.h"
#include "meshview/meshview.hpp"
#include "meshview/meshview_imgui.hpp"
#include "smplx/sequence.hpp"
#include "smplx/smplx.hpp"
#include "smplx/util.hpp"

using namespace smplx;

template <class ModelConfig>
static int run(std::string path) {
    SequenceAMASS amass(path);
    Gender gender = amass.gender;

    // * Construct SMPL body model
    Model<ModelConfig> model(gender);
    Body<ModelConfig> body(model);

    if (amass.n_frames) {
        // If not empty, load shape/pose
        amass.set_shape(body);
        amass.set_pose(body, 0);
    }
    body.update();

    // * Set up meshview viewer
    meshview::Viewer viewer;

    auto& smpl_mesh =
        viewer.add_mesh(body.verts(), model.faces, 0.8f, 0.5f, 0.6f);
    // Due to different coordinate system, all AMASS dada are rotated 90 degs
    // CCW on x-axis; we undo this rotation using the model matrix
    smpl_mesh.rotate(
        Eigen::AngleAxisf(M_PI * .5f, Eigen::Vector3f(-1.f, 0.f, 0.f))
            .toRotationMatrix());

    viewer.draw_axes = true;  // Press a to hide axes
    auto center_camera_on_human = [&]() {
        // Set camera's center of rotation to transformed root joint
        viewer.camera.center_of_rot = (/* model matrix */ smpl_mesh.transform *
                                       /* deformed root joint */ body.joints()
                                           .row(0)
                                           .transpose()
                                           .homogeneous())
                                          .template head<3>();
    };
    viewer.camera.dist_to_center = 4.f;  // Zoom out a little
    center_camera_on_human();

    // * Animation state
    int frame = 0;        // Current frame
    int frame_start = 0;  // Frame when we started animating
    std::chrono::high_resolution_clock::time_point
        time_start;  // Time when we started anim
    bool playing = false;
    bool camera_follow_human =
        true;  // Whether to automatically follow the human

    bool updated;

    // Copy AMASS frame 'frame' to body and update mesh + (optionally) camera
    auto update_frame = [&]() {
        if (amass.n_frames == 0) return;  // Empty sequence
        amass.set_pose(body, (size_t)frame);
        body.update();
        smpl_mesh.verts_pos().noalias() = body.verts();
        smpl_mesh.faces.noalias() = model.faces;
        if (camera_follow_human) {
            // Follow the human with camera (set c.o.r. to root joint)
            center_camera_on_human();
        }
        updated = true;
    };

    // Helper to toggle play/pause
    auto toggle_play = [&]() {
        playing = !playing;
        if (playing) {
            frame = frame_start;
            time_start = std::chrono::high_resolution_clock::now();
        } else {
            frame_start = frame;
        }
        viewer.loop_wait_events = !playing;
    };

    // Set key handler
    viewer.on_key = [&](int key, meshview::input::Action action,
                        int mods) -> bool {
        if (action == meshview::input::Action::press) {
            if (key == 'R') {
                frame_start = 0;
                playing = false;
                update_frame();
            } else if (key == 'L') {
                camera_follow_human = !camera_follow_human;
            } else if (key == ' ') {
                toggle_play();
            }
        }
        return true;
    };
    viewer.on_open = []() { ImGui::GetIO().IniFilename = nullptr; };
    viewer.on_loop = [&]() {
        updated = false;
        if (playing) {
            double delta =
                std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - time_start)
                    .count();
            int nx_frame =
                static_cast<int>(std::floor(delta * amass.frame_rate)) +
                frame_start;
            if (nx_frame > frame) {
                if (nx_frame >= amass.n_frames) {
                    playing = false;
                    frame_start = 0;
                } else {
                    frame = nx_frame;
                    update_frame();
                }
            }
        }
        return updated;
    };
    viewer.on_gui = [&]() {
        updated = false;
        static ImGui::FileBrowser open_file_dialog;
        if (open_file_dialog.GetTitle().empty()) {
            open_file_dialog.SetTypeFilters({".npz"});
            open_file_dialog.SetTitle("Open AMASS npz");
        }
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(300, 180), ImGuiCond_Once);
        ImGui::Begin("Control", NULL);
        ImGui::Text("Model: %s  Gender: %s", model.name(),
                    util::gender_to_str(model.gender));
        if (amass.n_frames) {
            ImGui::TextWrapped("Seq: %s", path.c_str());
            ImGui::Text("Frame %i (%i total)", frame, (int)amass.n_frames);
            if (ImGui::SliderInt("Frame##framectl", &frame, 0,
                                 (int)amass.n_frames - 1)) {
                frame_start = frame;
                if (playing)
                    time_start = std::chrono::high_resolution_clock::now();
                update_frame();
            }
            if (ImGui::Button(playing ? "Pause" : "Play")) {
                toggle_play();
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset")) {
                frame = frame_start = 0;
                playing = false;
                update_frame();
            }
            ImGui::SameLine();
        } else {
            ImGui::TextWrapped(
                "Please click 'Open AMASS npz' and select a file. "
                "Ignore the current camera angle, it will be correct on open.");
        }
        if (ImGui::Button("Open AMASS npz")) {
            frame = frame_start = 0;
            playing = false;
            // Need to disable glfwWaitEvents because
            // it makes the dialog unclickable
            viewer.loop_wait_events = false;
            open_file_dialog.Open();
        }
        ImGui::Checkbox("Camera follows human", &camera_follow_human);
        ImGui::End();  // Control

        open_file_dialog.Display();
        if (open_file_dialog.HasSelected()) {
            // Load new sequence
            path = open_file_dialog.GetSelected().string();
            amass.load(path);
            amass.set_shape(body);

            if (amass.gender != gender) {
                // Have to change the gender
                model.load(amass.gender);
                gender = amass.gender;
            }
            open_file_dialog.ClearSelected();
            update_frame();
            viewer.loop_wait_events = true;
        }
        return updated;
    };
    viewer.show();
    return 0;
}

int main(int argc, char** argv) {
    std::string path = argc > 2 ? argv[2] : "";
    if (argc < 2 || std::toupper(argv[1][0]) == 'H') {
        return run<model_config::SMPLH>(path);
    } else if (std::toupper(argv[1][0]) == 'S') {
        return run<model_config::SMPL>(path);
    } else if (std::toupper(argv[1][0]) == 'X') {
        return run<model_config::SMPLX>(path);
    }
}
