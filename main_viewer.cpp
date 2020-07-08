// Renders SMPL-X model in a OpenGL 3D viewer
// 2 optional arguments:
// 1. model type, default S
//    options: S H X (SMPL SMPL-H SMPL-X)
// 2. model gender, default NEUTRAL
//    options: NEUTRAL MALE FEMALE (case insensitive)
// Will load data/models/smplx/SMPLX_[arg].npz;
#include <iostream>
#include <algorithm>

#include "smplx/smplx.hpp"
#include "smplx/util.hpp"
#include "meshview/viewer.hpp"

#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"

using namespace smplx;

template<class ModelConfig>
static int run(Gender gender) {
    // * Construct SMPL body model
    Model<ModelConfig> model(gender);
    Body<ModelConfig> body(model);
    {
_SMPLX_BEGIN_PROFILE;
    body.update();
_SMPLX_PROFILE(UPDATE);
    }

    // * Set up meshview viewer
    meshview::Viewer viewer;
    // viewer.add(meshview::Mesh::Cube(0.5f)).translate(Eigen::Vector3f(0.f, -1.8f, 0.f))
    //     .set_shininess(32.f)
    //     .add_texture<meshview::Texture::TYPE_DIFFUSE>(
    //             util::find_data_file("tex/container2.png"))
    //     .add_texture<meshview::Texture::TYPE_SPECULAR>(
    //             util::find_data_file("tex/container2_specular.png"))
    //     ;

    {
        meshview::Mesh tmp_mesh(ModelConfig::n_verts(), ModelConfig::n_faces());
        tmp_mesh.verts_pos() = body.verts();
        tmp_mesh.faces = model.faces;
        viewer.add(std::move(tmp_mesh)).estimate_normals().set_shininess(4.f)
            .add_texture_solid<>(1.f, 0.7f, 0.8f)
            .add_texture_solid<meshview::Texture::TYPE_SPECULAR>(0.1f, 0.1f, 0.1f);
    }
    {
        meshview::PointCloud tmp_pc(ModelConfig::n_verts());
        Points colors(model.n_joints(), 3);
        for (size_t i = 0; i < model.n_joints(); ++i) {
            colors.row(i) = util::color_from_int(i).transpose();
        }
        tmp_pc.verts_pos().noalias() = body.verts();
        tmp_pc.verts_rgb().noalias() = model.weights * colors;
        viewer.add(std::move(tmp_pc)).translate(Eigen::Vector3f(2.f, 0.f, 0.f));
    }

    auto& smpl_mesh = viewer.meshes.back();
    auto& smpl_pc = viewer.point_clouds.back();

    auto update = [&model, &body, &smpl_mesh, &smpl_pc, &viewer]() {
_SMPLX_BEGIN_PROFILE;
        body.update();
_SMPLX_PROFILE(UPDATE);
        smpl_mesh.verts_pos() = body.verts();
        smpl_mesh.faces = model.faces;
        smpl_mesh.update();
        smpl_pc.verts_pos() = body.verts();
        smpl_pc.update();
    };

    viewer.on_open = [](){ ImGui::GetIO().IniFilename = nullptr; };
    viewer.on_gui = [&]() {
        // * GUI code
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(500, 360), ImGuiCond_Once);
        ImGui::Begin("Model Parameters", NULL);
        ImGui::Text("Model: %s  Gender: %s", model.name(), util::gender_to_str(model.gender));
        ImGui::TextUnformatted("Press h for help");
        ImGui::TextUnformatted("Reset: ");
        ImGui::SameLine();
        if (ImGui::Button("Trans##ResetTrans")) { body.trans().setZero(); update(); }
        ImGui::SameLine();
        if (ImGui::Button("Pose##ResetPose")) { body.pose().setZero(); update(); }
        ImGui::SameLine();
        if (ImGui::Button("Hand##ResetHand")) { body.hand_pca().setZero(); update(); }
        ImGui::SameLine();
        if (ImGui::Button("Shape##ResetShape")) { body.shape().setZero(); update(); }

        if(ImGui::SliderFloat3("translation", body.trans().data(), -5.f, 5.f)) update();
        if (ImGui::TreeNode("Pose")) {
            const int STEP = 10;
            for (size_t j = 0; j < model.n_explicit_joints(); j += STEP) {
                size_t end_idx = std::min(j + STEP, model.n_explicit_joints());
                if (ImGui::TreeNode(("Angle axis " + std::to_string(j) + " - " +
                            std::to_string(end_idx-1)).c_str())) {
                    for (size_t i = j; i < end_idx; ++i) {
                        if(ImGui::SliderFloat3((std::string(
                                            std::string(model.joint_name(i)) +
                                            "##joint") + std::to_string(i)).c_str(),
                                    body.pose().data() + i * 3, -1.6f, 1.6f)) update();
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
                                (std::string("pca_l") + std::to_string(i)).c_str(),
                                body.hand_pca_l().data() + i, -5.f, 5.f))
                            update();
                    }
                    ImGui::TreePop();
                }
                if (ImGui::TreeNode("Right Hand")) {
                    for (size_t i = 0; i < model.n_hand_pca(); ++i) {
                        if (ImGui::SliderFloat(
                                (std::string("pca_r") + std::to_string(i)).c_str(),
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
                if(ImGui::SliderFloat((std::string("shape") + std::to_string(i)).c_str(),
                            body.shape().data() + i, -5.f, 5.f)) update();
            }
            ImGui::TreePop();
        }
        ImGui::End(); // Model Parameters

        ImGui::SetNextWindowPos(ImVec2(10, 395), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(500, 100), ImGuiCond_Once);
        ImGui::Begin("Camera", NULL);

        if (ImGui::Button("Reset view")) {
            viewer.camera.reset_view();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset projection")) {
            viewer.camera.reset_proj();
        }

        if (ImGui::TreeNode("View")) {
            if(ImGui::SliderFloat3("cor##CORSLIDER", viewer.camera.center_of_rot.data(), -5.f, 5.f))
                viewer.camera.update_view();
            if(ImGui::SliderFloat("radius", &viewer.camera.dist_to_center, 0.01f, 5.f))
                viewer.camera.update_view();
            if(ImGui::DragFloat("yaw", &viewer.camera.yaw)) viewer.camera.update_view();
            if(ImGui::DragFloat("pitch", &viewer.camera.pitch)) viewer.camera.update_view();
            if(ImGui::DragFloat("roll", &viewer.camera.roll)) viewer.camera.update_view();
            if(ImGui::SliderFloat3("world_up", viewer.camera.world_up.data(), -5.f, 5.f))
                viewer.camera.update_view();
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Projection")) {
            if(ImGui::SliderFloat("fovy", &viewer.camera.fovy, 0.01f, 1.5f)) viewer.camera.update_proj();
            if(ImGui::SliderFloat("z_close", &viewer.camera.z_close, 0.01f, 10.f)) viewer.camera.update_proj();
            if(ImGui::SliderFloat("z_far", &viewer.camera.z_far, 11.f, 5000.f)) viewer.camera.update_proj();
            ImGui::TreePop();
        }

        ImGui::End(); // Model Parameters
    };
    viewer.show();
    return 0;
}

int main(int argc, char** argv) {
    Gender gender = util::parse_gender(argc > 2 ? argv[2] : "NEUTRAL");
    if (argc < 2 || std::toupper(argv[1][0]) == 'S') {
        return run<model_config::SMPL>(gender);
    } else if (std::toupper(argv[1][0]) == 'H') {
        return run<model_config::SMPLH>(gender);
    } else if (std::toupper(argv[1][0]) == 'X') {
        return run<model_config::SMPLX>(gender);
    }
}
