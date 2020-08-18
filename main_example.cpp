// Very basic example:
// Writes SMPL-X model to out.obj
// 1 optional argument: SMPL-X model gender, default NEUTRAL
// options: NEUTRAL MALE FEMALE (case insensitive)
// Will load data/models/smplx/SMPLX_[arg].npz;
// See include/smplx/model_config.hpp for joint names
#include "smplx/smplx.hpp"
#include "smplx/util.hpp"
#include <iostream>
int main(int argc, char** argv) {
    // ModelX/BodyX means SMPL-X model; *S is for SMPL, *H is for SMPL+H, *Xpca
    // is SMPL-X with hand pca you may provide a path to the .npz model instead
    // of a gender to the ModelX constructor
    smplx::ModelX model(
        smplx::util::parse_gender(argc > 1 ? argv[1] : "NEUTRAL"));
    smplx::BodyX body(model);
    // X axis rotation of r-knee
    body.pose()(3 * /*r knee*/ 5) = 0.5f;
    srand((unsigned)time(NULL));
    _SMPLX_BEGIN_PROFILE;
    body.update();
    _SMPLX_PROFILE(update time);
    body.save_obj("out.obj");
    std::cout << "Wrote to out.obj\n";
}
