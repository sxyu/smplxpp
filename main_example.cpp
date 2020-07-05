// Writes SMPL-X model to a file
#include "smpl/smpl.hpp"
#include "smpl/util.hpp"
int main(int argc, char** argv) {
    // 1 optional argument: SMPL-X model gender, default NEUTRAL
    // options: NEUTRAL MALE FEMALE
    // Will load data/models/smplx/SMPLX_[arg].npz;
    std::string model_name = argc > 1 ? argv[1] : "NEUTRAL";
    smpl::Model model(smpl::util::find_data_file("models/smplx/SMPLX_" + model_name + ".npz"),
                      smpl::util::find_data_file("models/smplx/uv.txt"));
    smpl::Body body(model);
    body.update();
    body.save_obj("out.obj");
}
