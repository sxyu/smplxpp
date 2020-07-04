#include <cstdint>
#include <iostream>

#include "smpl.hpp"
#include "util.hpp"
#include "viewer/viewer.hpp"
;
int32_t main(int argc, char** argv) {
    if (argc > 2) {
        std::cout << "Expected <= 1 argument: SMPL-X model gender\n"
            "Will load data/models/smplx/SMPLX_[arg].npz\n"
            "(default MALE)";
        return 1;
    }
    std::string model_name = argc > 1 ? argv[1] : "MALE";
    for (auto& c : model_name) c = std::toupper(c);

    using namespace smpl;

    std::cout << "Load SMPLX_" << model_name << "\n";
    Model model(util::find_data_file("models/smplx/SMPLX_" + model_name + ".npz"),
                util::find_data_file("models/smplx/uv.txt"));
    Body body(model);
    body.update();

    body.save_obj("test.obj");
    // Viewer viewer;
    // viewer.spin();

    return 0;
}
