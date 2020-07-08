#include <fstream>
#include <iomanip>
#include <iostream>

#include "smplx/smplx.hpp"
#include "smplx/util.hpp"
#include "smplx/internal/local_to_global.hpp"

namespace smplx {

template<class ModelConfig>
Body<ModelConfig>::Body(const Model<ModelConfig>& model, bool set_zero)
    : model(model), params(model.n_params()) {
    if (set_zero) this->set_zero();
    // Point cloud after applying shape keys but before lbs (num points, 3)
    _verts_shaped.resize(model.n_verts(), 3);

    // Joints after applying shape keys but before lbs (num joints, 3)
    _joints_shaped.resize(model.n_joints(), 3);

    // Final deformed point cloud
    _verts.resize(model.n_verts(), 3);

    // Affine joint transformation, as 3x4 matrices stacked horizontally (bottom
    // row omitted) NOTE: col major
    _joint_transforms.resize(model.n_joints(), 12);
#ifdef SMPLX_CUDA_ENABLED
    _cuda_load();
#endif
}

template<class ModelConfig>
Body<ModelConfig>::~Body() {
#ifdef SMPLX_CUDA_ENABLED
    _cuda_free();
#endif
}

template<class ModelConfig>
const Points& Body<ModelConfig>::verts() const {
#ifdef SMPLX_CUDA_ENABLED
    if (_last_update_used_gpu) _cuda_maybe_retrieve_verts();
#endif
    return _verts;
}
template<class ModelConfig>
const Points& Body<ModelConfig>::joints() const {
    return _joints;
}

// Main LBS routine
template<class ModelConfig>
void Body<ModelConfig>::update(bool force_cpu, bool enable_pose_blendshapes) {
    _SMPLX_BEGIN_PROFILE;
    // Will store full pose params (angle-axis), including hand
    Vector full_pose(3 * model.n_joints());

    // Shape params +/ linear joint transformations as flattened 3x3 rotation
    // matrices rowmajor, only for blend shapes
    Vector blendshape_params(model.n_blend_shapes());

    using AffineTransformMap =
        Eigen::Map<Eigen::Matrix<Scalar, 3, 4, Eigen::RowMajor> >;
    using RotationMap =
        Eigen::Map<Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> >;

    // Copy body pose onto full pose
    full_pose.head(3 * model.n_explicit_joints()).noalias() = pose();
    if (model.n_hand_pca_joints() > 0) {
        // Use hand PCA weights to fill in hand pose within full pose
        full_pose.segment(3 * model.n_explicit_joints(), 3 * model.n_hand_pca_joints())
            .noalias() = model.hand_mean_l + model.hand_comps_l * hand_pca_l();
        full_pose.tail(3 * model.n_hand_pca_joints()).noalias() =
            model.hand_mean_r + model.hand_comps_r * hand_pca_r();
    }

    // Copy shape params to blendshape params
    blendshape_params.head<ModelConfig::n_shape_blends()>() = shape();

    // Convert angle-axis to rotation matrix using rodrigues
    AffineTransformMap(_joint_transforms.topRows<1>().data())
        .template leftCols<3>().noalias() =
        util::rodrigues<float>(full_pose.head<3>());
    for (size_t i = 1; i < model.n_joints(); ++i) {
        AffineTransformMap joint_trans(_joint_transforms.row(i).data());
        joint_trans.template leftCols<3>().noalias() =
            util::rodrigues<float>(full_pose.segment<3>(3 * i));
        RotationMap mp(blendshape_params.data() +  9 * i + (model.n_shape_blends() - 9));
        mp.noalias() = joint_trans.template leftCols<3>();
        mp.diagonal().array() -= 1.f;
    }

#ifdef SMPLX_CUDA_ENABLED
    _last_update_used_gpu = !force_cpu;
    if (!force_cpu) {
        _cuda_update(blendshape_params.data(),
                     _joint_transforms.data(),
                     enable_pose_blendshapes);
        return;
    }
#endif

    // _SMPLX_PROFILE(preproc);
    // Apply blend shapes
    {
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > verts_shaped_flat(
            _verts_shaped.data(), model.n_verts() * 3);
        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> >
            verts_init_flat(model.verts.data(), model.n_verts() * 3);
        if (enable_pose_blendshapes) {
            // HORRIBLY SLOW, like 95% of the time is spent here yikes
            // Add shape blend shapes
            verts_shaped_flat.noalias() = verts_init_flat +
                model.blend_shapes * blendshape_params;
        } else {
            // Add shape blend shapes
            verts_shaped_flat.noalias() = verts_init_flat +
                    model.blend_shapes.template leftCols<ModelConfig::n_shape_blends()>() *
                    blendshape_params.head<ModelConfig::n_shape_blends()>();
        }
    }
    // _SMPLX_PROFILE(blendshape);

    // Apply joint regressor
    _joints_shaped = model.joint_reg * _verts_shaped;

    local_to_global<ModelConfig>(trans(), _joints_shaped, _joints, _joint_transforms);
    // _SMPLX_PROFILE(localglobal);

    // * LBS *
    // Construct a transform for each vertex
    Eigen::Matrix<Scalar, Eigen::Dynamic, 12, Eigen::RowMajor> vert_transforms =
        model.weights * _joint_transforms;
    // _SMPLX_PROFILE(lbs weight computation);

    // Apply affine transform to each vertex and store to output
    for (size_t i = 0; i < model.n_verts(); ++i) {
        AffineTransformMap transform(vert_transforms.row(i).data());
        _verts.row(i) =
            _verts_shaped.row(i).homogeneous() * transform.transpose();
    }
    // _SMPLX_PROFILE(lbs point transform);
}

template<class ModelConfig>
void Body<ModelConfig>::save_obj(const std::string& path) const {
    const auto& cur_verts = verts();
    if (cur_verts.rows() == 0) return;
    std::ofstream ofs(path);
    ofs << "# Generated by SMPL-X_cpp"
        << "\n";
    ofs << std::fixed << std::setprecision(6) << "o smplx\n";
    for (int i = 0; i < model.n_verts(); ++i) {
        ofs << "v " << cur_verts(i, 0) << " " << cur_verts(i, 1) << " "
            << cur_verts(i, 2) << "\n";
    }
    ofs << "s 1\n";
    for (int i = 0; i < model.n_faces(); ++i) {
        ofs << "f " << model.faces(i, 0) + 1 << " " << model.faces(i, 1) + 1
            << " " << model.faces(i, 2) + 1 << "\n";
    }
    ofs.close();
}

// Instantiation
template class Body<model_config::SMPL>;
template class Body<model_config::SMPLH>;
template class Body<model_config::SMPLX>;

}  // namespace smplx
