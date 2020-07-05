// Shows SMPL-X model
#include <iostream>
#include <algorithm>

#include "smpl/smpl.hpp"
#include "smpl/util.hpp"
#include "meshview/viewer.hpp"

#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"

int main(int argc, char** argv) {
    // 1 optional argument: SMPL-X model gender, default NEUTRAL
    // options: NEUTRAL MALE FEMALE
    // Will load data/models/smplx/SMPLX_[arg].npz;
    std::string model_name = argc > 1 ? argv[1] : "NEUTRAL";

    // * Construct SMPL body model
    smpl::Model model(smpl::util::find_data_file("models/smplx/SMPLX_" + model_name + ".npz"),
                      smpl::util::find_data_file("models/smplx/uv.txt"));
    smpl::Body body(model); body.update();

    // * Set up meshview viewer
    meshview::Viewer viewer;

    {
        meshview::Mesh tmp_mesh(model.n_verts, model.n_faces);
        tmp_mesh.verts_pos() = body.verts;
        tmp_mesh.faces = model.faces;
        viewer.add(std::move(tmp_mesh)).estimate_normals().set_shininess(4.f)
            .add_texture_solid<>(1.f, 0.7f, 0.8f)
            .add_texture_solid<meshview::Texture::TYPE_SPECULAR>(0.1f, 0.1f, 0.1f);
    }
    meshview::Mesh& smpl_mesh = viewer.meshes.back();

    auto update = [&model, &body, &smpl_mesh, &viewer]() {
        body.update();
        smpl_mesh.verts_pos() = body.verts;
        smpl_mesh.faces = model.faces;
        smpl_mesh.update();
    };

    viewer.on_gui = [&]() {
        // * GUI code
        ImGui::Begin("Model Parameters", NULL);
        if(ImGui::SliderFloat3("position", body.trans().data(), -5.f, 5.f)) update();
        if (ImGui::TreeNode("Pose")) {
            const int STEP = 10;
            for (size_t j = 0; j < model.n_body_joints; j += STEP) {
                size_t end_idx = std::min(j + STEP, model.n_body_joints);
                if (ImGui::TreeNode(("Angle axis " + std::to_string(j) + " - " +
                            std::to_string(end_idx-1)).c_str())) {
                    for (size_t i = j; i < end_idx; ++i) {
                        if(ImGui::SliderFloat3((std::string("joint") + std::to_string(i)).c_str(),
                                    body.pose().data() + i * 3, -1.6f, 1.6f)) update();
                    }
                    ImGui::TreePop();
                }
            }
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Hand PCA")) {
            if (ImGui::TreeNode("Left Hand")) {
                for (size_t i = 0; i < model.n_hand_pca; ++i) {
                    if(ImGui::SliderFloat((std::string("pca_l") + std::to_string(i)).c_str(),
                                body.hand_pca_l().data() + i, -5.f, 5.f)) update();
                }
                ImGui::TreePop();
            }
            if (ImGui::TreeNode("Right Hand")) {
                for (size_t i = 0; i < model.n_hand_pca; ++i) {
                    if(ImGui::SliderFloat((std::string("pca_r") + std::to_string(i)).c_str(),
                                body.hand_pca_r().data() + i, -5.f, 5.f)) update();
                }
                ImGui::TreePop();
            }
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Shape")) {
            for (size_t i = 0; i < model.n_shape_blends; ++i) {
                if(ImGui::SliderFloat((std::string("shape") + std::to_string(i)).c_str(),
                            body.shape().data() + i, -5.f, 5.f)) update();
            }
            ImGui::TreePop();
        }
        ImGui::End(); // Model Parameters

        ImGui::Begin("Camera", NULL);
        // static Eigen::Vector3f camera_center = viewer.camera.set_center;
        // if(ImGui::SliderFloat("dist_to_cen", viewer.camera.center, 0.01f, 5.f)) update();

        ImGui::End(); // Model Parameters
    };
    viewer.show();

    return 0;
}
