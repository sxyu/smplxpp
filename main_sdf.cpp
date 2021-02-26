#include "smplx/smplx.hpp"
#include "smplx/util.hpp"
#include "meshview/meshview.hpp"
#include "meshview/meshview_imgui.hpp"
#include "sdf/sdf.hpp"

#include <iostream>
#include <cstddef>
#include <cstdint>
#include <cmath>

using namespace smplx;

template <class ModelConfig>
static int run(Gender gender, bool robust) {
    Model<ModelConfig> model(gender);
    Body<ModelConfig> body(model);
    body.update();

    sdf::SDF sdf(body.verts(), model.faces, robust);

    meshview::Viewer viewer;
    viewer.draw_axes = false;
    viewer.wireframe = true;

    // Generate flat point cloud, for visualizing a cross-section of the SDF
    const int FLAT_CLOUD_DIM = 400;
    const float FLAT_CLOUD_RADIUS_X = 1.0, FLAT_CLOUD_RADIUS_Y = 1.4;
    const float FLAT_CLOUD_STEP_X = FLAT_CLOUD_RADIUS_X * 2 / FLAT_CLOUD_DIM;
    const float FLAT_CLOUD_STEP_Y = FLAT_CLOUD_RADIUS_Y * 2 / FLAT_CLOUD_DIM;
    Points _pts_flat(FLAT_CLOUD_DIM * FLAT_CLOUD_DIM, 3);

    for (int i = 0; i < FLAT_CLOUD_DIM; ++i) {
        float y = -FLAT_CLOUD_RADIUS_Y + FLAT_CLOUD_STEP_Y * i;
        for (int j = 0; j < FLAT_CLOUD_DIM; ++j) {
            float x = -FLAT_CLOUD_RADIUS_X + FLAT_CLOUD_STEP_X * j;
            _pts_flat.row(i * FLAT_CLOUD_DIM + j) << x, y, 0.f;
        }
    }

    // Add planar cross section point cloud
    auto& flat_cloud = viewer.add_point_cloud(_pts_flat, 0.0, 1.0, 0.0);
    auto pts_flat = flat_cloud.verts_pos();

    // Add body skin points as mesh
    auto& smpl_mesh = viewer.add_mesh(body.verts(), model.faces, 1.0, 1.0, 1.0);

    float flat_z = 0.0f;
    const float MAX_DISTANCE_FUNC = 0.09f;

    // Color by containment only
    bool contains_only = false;

    // Update the flat (cross section) point cloud
    auto update_flat_cloud = [&]() {
        pts_flat.rightCols<1>().setConstant(flat_z);
        Eigen::VectorXf sdf_flat;
        _SMPLX_BEGIN_PROFILE;
        sdf_flat.noalias() = sdf(pts_flat);
        _SMPLX_PROFILE(compute SDF);
        for (size_t i = 0; i < pts_flat.rows(); ++i) {
            float t = contains_only ? 1.f
                                    : (1.f - std::min(std::abs(sdf_flat[i]),
                                                      MAX_DISTANCE_FUNC) *
                                                 (1.f / MAX_DISTANCE_FUNC));

            auto rgb = flat_cloud.verts_rgb().row(i);
            rgb[0] = (sdf_flat[i] < 0) ? 0.0f : 1.0f;
            rgb[1] = t;
            rgb[2] = t * 0.5;
        }
    };
    update_flat_cloud();

    bool updated = false;

    // Update smpl and flat point cloud
    auto update = [&]() {
        body.update();
        smpl_mesh.verts_pos().noalias() = body.verts();
        sdf.update();
        update_flat_cloud();
        // Update the mesh on-the-fly (send to GPU)
        updated = true;
    };

    viewer.on_key = [&](int key, meshview::input::Action action,
                        int mods) -> bool {
        if (action != meshview::input::Action::release) {
            if (key == 'J' || key == 'K') {
                flat_z += (key == 'J') ? 0.01 : -0.01;
                std::cout << "z = " << flat_z << "\n";
                update_flat_cloud();
                flat_cloud.update();
            }
        }
        return true;
    };

    viewer.on_open = []() { ImGui::GetIO().IniFilename = nullptr; };
    viewer.on_gui = [&]() {
        updated = false;
        // * GUI code
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(500, 360), ImGuiCond_Once);
        ImGui::Begin("Model and Cross Section", NULL);
        ImGui::Text("Model: %s  Gender: %s", model.name(),
                    util::gender_to_str(model.gender));
        ImGui::TextUnformatted("Press h for help");
        ImGui::TextUnformatted("Reset: ");
        ImGui::SameLine();
        if (ImGui::Button("Pose##ResetPose")) {
            body.pose().setZero();
            update();
        }
        ImGui::SameLine();
        if (ImGui::Button("Hand##ResetHand")) {
            body.hand_pca().setZero();
            update();
        }
        ImGui::SameLine();
        if (ImGui::Button("Shape##ResetShape")) {
            body.shape().setZero();
            update();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cross Sec##ResetCrossSection")) {
            flat_z = 0.f;
            update();
        }

        ImGui::Checkbox("show mesh", &smpl_mesh.enabled);
        if (ImGui::Checkbox("containment only", &contains_only)) {
            update_flat_cloud();
            updated = true;
        }

        if (ImGui::SliderFloat("cross sec z##slideflatz", &flat_z, -1.0f,
                               1.0f)) {
            update();
        }
        ImGui::TextUnformatted("Tip: press j,k to adjust cross section");

        if (ImGui::TreeNode("Pose")) {
            const int STEP = 10;
            for (size_t j = 0; j < model.n_explicit_joints(); j += STEP) {
                size_t end_idx = std::min(j + STEP, model.n_explicit_joints());
                if (ImGui::TreeNode(("Angle axis " + std::to_string(j) + " - " +
                                     std::to_string(end_idx - 1))
                                        .c_str())) {
                    for (size_t i = j; i < end_idx; ++i) {
                        if (ImGui::SliderFloat3(
                                (std::string(std::string(model.joint_name(i)) +
                                             "##joint") +
                                 std::to_string(i))
                                    .c_str(),
                                body.pose().data() + i * 3, -1.6f, 1.6f))
                            update();
                    }
                    ImGui::TreePop();
                }
            }
            ImGui::TreePop();
        }
        if (model.n_hand_pca()) {
            if (ImGui::TreeNode("Hand PCA")) {
                if (ImGui::TreeNode("Left Hand")) {
                    for (size_t i = 0; i < model.n_hand_pca(); ++i) {
                        if (ImGui::SliderFloat(
                                (std::string("pca_l") + std::to_string(i))
                                    .c_str(),
                                body.hand_pca_l().data() + i, -5.f, 5.f))
                            update();
                    }
                    ImGui::TreePop();
                }
                if (ImGui::TreeNode("Right Hand")) {
                    for (size_t i = 0; i < model.n_hand_pca(); ++i) {
                        if (ImGui::SliderFloat(
                                (std::string("pca_r") + std::to_string(i))
                                    .c_str(),
                                body.hand_pca_r().data() + i, -5.f, 5.f))
                            update();
                    }
                    ImGui::TreePop();
                }
                ImGui::TreePop();
            }
        }
        if (ImGui::TreeNode("Shape")) {
            for (size_t i = 0; i < model.n_shape_blends(); ++i) {
                if (ImGui::SliderFloat(
                        (std::string("shape") + std::to_string(i)).c_str(),
                        body.shape().data() + i, -5.f, 5.f))
                    update();
            }
            ImGui::TreePop();
        }
        ImGui::End();  // Model Parameters

        ImGui::SetNextWindowPos(ImVec2(10, 395), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(500, 100), ImGuiCond_Once);
        ImGui::Begin("Camera and Rendering", NULL);

        if (ImGui::Button("Reset view")) {
            viewer.camera.reset_view();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset projection")) {
            viewer.camera.reset_proj();
        }
        ImGui::SameLine();
        ImGui::Checkbox("wireframe", &viewer.wireframe);

        if (ImGui::TreeNode("View")) {
            if (ImGui::SliderFloat3("cen_of_rot",
                                    viewer.camera.center_of_rot.data(), -5.f,
                                    5.f))
                viewer.camera.update_view();
            if (ImGui::SliderFloat("radius", &viewer.camera.dist_to_center,
                                   0.01f, 10.f))
                viewer.camera.update_view();
            if (ImGui::DragFloat("yaw", &viewer.camera.yaw))
                viewer.camera.update_view();
            if (ImGui::DragFloat("pitch", &viewer.camera.pitch))
                viewer.camera.update_view();
            if (ImGui::DragFloat("roll", &viewer.camera.roll))
                viewer.camera.update_view();
            if (ImGui::SliderFloat3("world_up", viewer.camera.world_up.data(),
                                    -5.f, 5.f))
                viewer.camera.update_view();
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Projection")) {
            if (ImGui::SliderFloat("fovy", &viewer.camera.fovy, 0.01f, 1.5f))
                viewer.camera.update_proj();
            if (ImGui::SliderFloat("z_close", &viewer.camera.z_close, 0.01f,
                                   10.f))
                viewer.camera.update_proj();
            if (ImGui::SliderFloat("z_far", &viewer.camera.z_far, 11.f, 5000.f))
                viewer.camera.update_proj();
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Lighting")) {
            if (ImGui::SliderFloat3("pos", viewer.light_pos.data(), -4.f, 4.f))
                if (ImGui::SliderFloat3(
                        "ambient", viewer.light_color_ambient.data(), 0.f, 1.f))
                    if (ImGui::SliderFloat3("diffuse",
                                            viewer.light_color_diffuse.data(),
                                            0.f, 1.f))
                        if (ImGui::SliderFloat3(
                                "specular", viewer.light_color_specular.data(),
                                0.f, 1.f))
                            ImGui::TreePop();
        }

        ImGui::End();  // Camera and Rendering
        // Return true if updated to indicate mesh data has been changed
        // the viewer will update the GPU buffers automatically
        return updated;
    };
    viewer.show();
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Please enter model type S/H/X/Z/Xp/Zp\n";
    }
    Gender gender = util::parse_gender(argc > 2 ? argv[2] : "NEUTRAL");
    bool robust = argc > 3 ? (std::string(argv[3]) != "no") : true;

    std::string model_name = argv[1];
    for (auto& c : model_name) c = std::toupper(c);
    if (argc < 2 || model_name == "S") {
        return run<model_config::SMPL>(gender, robust);
    } else if (model_name == "H") {
        return run<model_config::SMPLH>(gender, robust);
    } else if (model_name == "X") {
        return run<model_config::SMPLX>(gender, robust);
    } else if (model_name == "Z") {
        return run<model_config::SMPLX_v1>(gender, robust);
    } else if (model_name == "Xp") {
        return run<model_config::SMPLXpca>(gender, robust);
    } else if (model_name == "Zp") {
        return run<model_config::SMPLXpca_v1>(gender, robust);
    }
}
