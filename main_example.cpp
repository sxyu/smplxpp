// Writes SMPL-X model to out.obj
// 1 optional argument: SMPL-X model gender, default NEUTRAL
// options: NEUTRAL MALE FEMALE (case insensitive)
// Will load data/models/smplx/SMPLX_[arg].npz;
#include "smplx/smplx.hpp"
#include "smplx/util.hpp"
#include <iostream>
int main(int argc, char** argv) {
    smplx::ModelS model(smplx::util::parse_gender(argc > 1 ? argv[1] : "NEUTRAL"));
    smplx::BodyS body(model);
    srand(time(NULL));
    _SMPLX_BEGIN_PROFILE;
    body.update();
    const auto& verts = body.verts();
    _SMPLX_PROFILE(update + transfer time);
    body.save_obj("out.obj");
}
