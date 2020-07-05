#include <cstdint>
#include <iostream>

#include "smpl/smpl.hpp"
#include "smpl/util.hpp"
#include "meshview/viewer.hpp"

float vertices[] = {
    -0.5f, -0.5f, 0.0f,
    0.5f, -0.5f, 0.0f,
    0.0f,  0.5f, 0.0f
};

int32_t main(int argc, char** argv) {
    if (argc > 2) {
        std::cout << "Expected <= 1 argument: SMPL-X model gender\n"
            "Will load data/models/smplx/SMPLX_[arg].npz\n"
            "(default MALE)";
        return 1;
    }
    std::string model_name = argc > 1 ? argv[1] : "MALE";
    for (auto& c : model_name) c = std::toupper(c);

    using namespace meshview;

    std::cout << "Load SMPLX_" << model_name << "\n";
    smpl::Model model(util::find_data_file("models/smplx/SMPLX_" + model_name + ".npz"),
                     util::find_data_file("models/smplx/uv.txt"));
    smpl::Body body(model);
    body.update();

    // body.save_obj("test.obj");
    Viewer viewer;
    Mesh mesh(model.n_verts, model.n_faces);
    mesh.verts_pos() = body.verts;
    // std::cout << body.verts.middleRows<15>(5500) << "\n\n";
    // std::cout << model.verts.middleRows<15>(5500) << "\n";
    mesh.faces = model.faces;
    viewer.scene.add(mesh).estimate_normals().set_shininess(4.f)
         .add_texture_solid<Texture::TYPE_DIFFUSE>(1.f, 0.7f, 0.8f)
         .add_texture_solid<Texture::TYPE_SPECULAR>(0.1f, 0.1f, 0.1f);
         // .scale(5.f) ;

    for (size_t i = 0; i < model.n_joints; ++i) {
        Mesh cube = Mesh::Cube(0.02f);
        viewer.scene.add(std::move(cube))
            .set_shininess(32.f)
            .add_texture_solid<Texture::TYPE_DIFFUSE>(0.7f, 0.8f, 0.8f)
            // .add_texture<Texture::TYPE_DIFFUSE>(util::find_data_file("tex/container2.png"))
            // .add_texture<Texture::TYPE_SPECULAR>(util::find_data_file("tex/container2_specular.png"))
            .translate(body.joints.row(i).transpose());
    }
    // std::cout << model.joints << "\nJNT\n";
    // viewer.scene.add(Mesh::Square(0.5f))
    //     .set_shininess(32.f)
    //     .add_texture<Texture::TYPE_DIFFUSE>(util::find_data_file("tex/container2.png"))
    //     .add_texture<Texture::TYPE_SPECULAR>(util::find_data_file("tex/container2_specular.png"))
    //     .translate(Vector3f(0,-1,0));
    viewer.wireframe = false;

    viewer.show();
    // viewer.show();

    return 0;
}
