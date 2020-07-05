// Writes SMPL-X model to a file
#include <cstdint>
#include <iostream>

#include "smpl/smpl.hpp"
#include "smpl/util.hpp"

int32_t main(int argc, char** argv) {
    if (argc > 2) {
        std::cout << "Expected <= 1 argument: SMPL-X model gender\n"
            "Will load data/models/smplx/SMPLX_[arg].npz\n"
            "(default NEUTRAL)";
        return 1;
    }
    std::string model_name = argc > 1 ? argv[1] : "NEUTRAL";
    for (auto& c : model_name) c = std::toupper(c);

    std::cout << "Load SMPLX_" << model_name << "\n";
    smpl::Model model(smpl::util::find_data_file("models/smplx/SMPLX_" + model_name + ".npz"),
                      smpl::util::find_data_file("models/smplx/uv.txt"));
    smpl::Body body(model);
    body.update();
    body.save_obj("out.obj");

    return 0;
}
