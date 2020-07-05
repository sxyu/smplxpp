#include "smpl/smpl.hpp"

#include "smpl/util.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace smpl {

Body::Body(const Model& model, bool set_zero) : model(model), data(model.n_params) {
    if (set_zero) this->set_zero();
}

void Body::set_zero() { data.setZero(); }

// Main LBS routine
void Body::update(bool enable_pose_blendshapes) {
    // _SMPL_BEGIN_PROFILE;
    // Will store full pose params (angle-axis), including hand
    Vector full_pose(3 * model.n_joints);

    /** Point cloud after applying shape keys but before lbs (num points, 3) */
    PointCloud verts_shaped(model.n_verts, 3);

    // Affine joint transformation, as 3x4 matrices stacked horizontally (bottom row omitted)
    // NOTE: col major
    Eigen::Matrix<Scalar, Eigen::Dynamic, 12, Eigen::RowMajor> joint_transforms(model.n_joints, 12);

    // Linear joint transformations, as flattened 3x3 rotation matrices rowmajor
    // only for pose-dep blend shapes
    Vector pose_rotation_flat(model.n_joints * 9 - 9);

    using AffineTransformMap = Eigen::Map<Eigen::Matrix<Scalar, 3, 4, Eigen::RowMajor> >;
    using RotationMap = Eigen::Map<Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> >;

    // _SMPL_PROFILE(alloc);

    pose().segment<3>(12) = Eigen::Vector3f(0.3f, 0.0f, 0.0f);
    // Copy body pose onto full pose
    full_pose.head(3 * model.n_body_joints).noalias() = pose();
    if (model.n_hand_joints > 0) {
        // full_pose.tail(6 * model.n_hand_joints).setZero();
        // Use hand PCA weights to fill in hand pose within full pose
        full_pose.segment(3 * model.n_body_joints, 3 * model.n_hand_joints).noalias() =
            model.hand_mean_l + model.hand_comps_l * hand_pca_l();
        full_pose.tail(3 * model.n_hand_joints).noalias() =
            model.hand_mean_r + model.hand_comps_r * hand_pca_r();
        // _SMPL_PROFILE(hand pca);
    }

    // Convert angle-axis to rotation matrix using rodrigues
    for (int i = 0; i < model.n_joints; ++i) {
        AffineTransformMap joint_trans(joint_transforms.row(i).data());
        joint_trans.template leftCols<3>().noalias()
            = util::rodrigues<float>(full_pose.segment<3>(3 * i));
        if (i) {
            RotationMap(pose_rotation_flat.data() + 9 * i - 9).noalias()
                = joint_trans.template leftCols<3>();
        }
    }
    // _SMPL_PROFILE(rodrigues);

    // Apply blend shapes
    {
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1> >
            verts_shaped_flat(verts_shaped.data(), model.n_verts * 3);
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> >
            verts_init_flat(model.verts.data(), model.n_verts * 3);
        // Add shape blend shapes
        verts_shaped_flat.noalias() = model.shape_blend * shape() + verts_init_flat;
        if (enable_pose_blendshapes) {
            // Add pose blend shapes
            // HORRIBLY SLOW, like 95% of the time is spent here yikes
            verts_shaped_flat.noalias() += model.pose_blend * pose_rotation_flat;
        }
        // _SMPL_PROFILE(blendshape);
    }

    {
        // Apply joint regressor
        joints_shaped.resize(model.n_joints, 3);
        joints_shaped = model.joint_reg * verts_shaped;

        // _SMPL_PROFILE(jointregr);
        if (deform.rows() == model.n_verts) {
            // Maybe apply deformation
            verts_shaped.noalias() += deform;
            // _SMPL_PROFILE(deform);
        }
    }

    joints.resize(model.n_joints, 3);
    // Complete the affine transforms for each joint by adding translation
    // components and composing with parent
    for (int i = 0; i < model.n_joints; ++i) {
        AffineTransformMap transform(joint_transforms.row(i).data());
        // Set translation
        auto p = model.parent[i];
        if (p == -1) {
            transform.template rightCols<1>().noalias()
                = joints_shaped.row(i).transpose() + trans();
        } else {
            transform.template rightCols<1>().noalias()
                = (joints_shaped.row(i) - joints_shaped.row(p)).transpose();
            util::mul_affine<float, Eigen::RowMajor>(
                    AffineTransformMap(joint_transforms.row(p).data()),
                    transform);
        }
        // Grab the joint position in case the user wants it
        joints.row(i).noalias() = transform.topRightCorner<3, 1>().transpose();
    }
    for (int i = 0; i < model.n_joints; ++i) {
        AffineTransformMap transform(joint_transforms.row(i).data());
        // Normalize the translation to global
        transform.rightCols<1>() -= transform.leftCols<3>() * joints_shaped.row(i).transpose();
    }

    // _SMPL_PROFILE(mkaffine);

    // * LBS *
    // Construct a transform for each vertex
    Eigen::Matrix<Scalar, Eigen::Dynamic, 12, Eigen::RowMajor> vert_transforms =
        model.weights * joint_transforms;
    // _SMPL_PROFILE(lbs weight);

    // Apply affine transform to each vertex and store to output
    verts.resize(model.n_verts, 3);
    for (size_t i = 0; i < model.n_verts; ++i) {
        AffineTransformMap transform(vert_transforms.row(i).data());
        verts.row(i) = verts_shaped.row(i).homogeneous() * transform.transpose();
    }
    // _SMPL_PROFILE(lbs main);
}

void Body::save_obj(const std::string& path) const {
    if (verts.rows() == 0) return;
    std::ofstream ofs(path);
    ofs << "# Generated by SMPL-X_cpp" << "\n";
    ofs << std::fixed << std::setprecision(6) << "o smplx\n";
    for (int i = 0; i < model.n_verts; ++i) {
        ofs << "v " << model.verts(i, 0) << " " << model.verts(i, 1) << " " << model.verts(i, 2) << "\n";
    }
    ofs << "s 1\n";
    for (int i = 0; i < model.n_faces; ++i) {
        ofs << "f " << model.faces(i, 0) + 1 << " " << model.faces(i, 1) + 1 << " " << model.faces(i, 2) + 1 << "\n";
    }
    ofs.close();
}

}  // namespace smpl
