#include <Eigen/Core>
#include "smplx/smplx.hpp"
#include "smplx/util.hpp"

namespace smplx {
namespace {
// Local to global coordinates
// trans: root translation (3-vector)
// in_joints_shaped: joint positions from regressor
// out_joints: where to output deformed joint positions (for user access)
// joint_transforms: input/output joint transform
template<class ModelConfig>
SMPLX_HOST inline void local_to_global(const Eigen::Ref<const Eigen::Vector3f>& trans,
        const Eigen::Ref<const Points>& in_joints_shaped,
        Points& out_joints,
        Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, 12, Eigen::RowMajor>> joint_transforms) {
    out_joints.resize(ModelConfig::n_joints(), 3);
    using AffineTransformMap = Eigen::Map<Eigen::Matrix<Scalar, 3, 4, Eigen::RowMajor> >;
    // Handle root joint transforms
    AffineTransformMap root_transform(joint_transforms.topRows<1>().data());
    root_transform.template rightCols<1>().noalias() =
        in_joints_shaped.topRows<1>().transpose() + trans;
    out_joints.template topRows<1>().noalias() = root_transform.template rightCols<1>().transpose();

    // Complete the affine transforms for all other joint by adding translation
    // components and composing with parent
    for (int i = 1; i < ModelConfig::n_joints(); ++i) {
        AffineTransformMap transform(joint_transforms.row(i).data());
        const auto p = ModelConfig::parent[i];
        // Set relative translation
        transform.template rightCols<1>().noalias() =
            (in_joints_shaped.row(i) - in_joints_shaped.row(p)).transpose();
        // Compose rotation with parent
        util::mul_affine<float, Eigen::RowMajor>(
            AffineTransformMap(joint_transforms.row(p).data()), transform);
        // Grab the joint position in case the user wants it
        out_joints.row(i).noalias() = transform.template rightCols<1>().transpose();
    }

    for (int i = 0; i < ModelConfig::n_joints(); ++i) {
        AffineTransformMap transform(joint_transforms.row(i).data());
        // Normalize the translation to global
        transform.template rightCols<1>().noalias() -=
            transform.template leftCols<3>() * in_joints_shaped.row(i).transpose();
    }
}
}  // namespace
}  // namespace smplx
