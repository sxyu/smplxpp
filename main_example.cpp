// Very basic example:
// Writes SMPL-X model to out.obj
// 1 optional argument: SMPL-X model gender, default NEUTRAL
// options: NEUTRAL MALE FEMALE (case insensitive)
// Will load data/models/smplx/SMPLX_[arg].npz;
#include "smplx/smplx.hpp"
#include "smplx/util.hpp"
#include <iostream>
int main(int argc, char** argv) {
    smplx::ModelX model(smplx::util::parse_gender(argc > 1 ? argv[1] : "NEUTRAL"));
    smplx::BodyX body(model);
    // X axis rotation of r-knee
    body.pose()(3 * /*r knee*/5) = 0.5f;
    // See include/smplx/model_config.hpp for joint names
    srand((unsigned) time(NULL));
    _SMPLX_BEGIN_PROFILE;
    body.update();
    const auto& verts = body.verts();
	_SMPLX_PROFILE(update + transfer time);
    body.save_obj("out.obj");
	std::cout << "Wrote to out.obj\n";
}
