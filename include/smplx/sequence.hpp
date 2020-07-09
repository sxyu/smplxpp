#pragma once
#ifndef SMPLX_SEQUENCE_9512D947_1D9B_478F_AAB5_6E6A846A6828
#define SMPLX_SEQUENCE_9512D947_1D9B_478F_AAB5_6E6A846A6828

#include "smplx/smplx.hpp"
#include "smplx/sequence_config.hpp"
namespace smplx {


template<class SequenceConfig> class Sequence;

namespace internal {
    // Sadly, C++ does not allow specialization of template classes,
    // so we have to implement in a separate struct;
    // Also, C++14 doesn't allow specializing member structs, so we have to put it here
    template<class SequenceConfig, class ModelConfig>
    struct SequenceModelSpec {
        // Set shape
        static void set_shape(const Sequence<SequenceConfig>& seq,
                Body<ModelConfig>& body) {
            throw std::invalid_argument(std::string(
                        "smplx::Sequence does not currently support model: ") +
                    ModelConfig::model_name);
        }
        // Set pose and root transform
        static void set_pose(const Sequence<SequenceConfig>& seq,
                Body<ModelConfig>& body, size_t frame) {
            throw std::invalid_argument(std::string(
                        "smplx::Sequence does not currently support model: ") +
                    ModelConfig::model_name);
        }
    };

    // Per-model specialization
    template <class SequenceConfig>
    struct SequenceModelSpec<SequenceConfig, model_config::SMPL> {
        static void set_shape(const Sequence<SequenceConfig>& seq,
                Body<model_config::SMPL>& body) {
            body.shape().noalias() = seq.shape
                .template head<model_config::SMPL::n_shape_blends()>();
        }
        static void set_pose(const Sequence<SequenceConfig>& seq,
                Body<model_config::SMPL>& body, size_t frame) {
            constexpr size_t n_common_joints = SequenceConfig::n_body_joints() * 3;
            body.trans().noalias() = seq.trans.row(frame).transpose();
            body.pose().template head<n_common_joints>().noalias() =
                seq.pose.row(frame).template head<n_common_joints>().transpose();
            body.pose()
                .template tail<model_config::SMPL::n_explicit_joints() * 3
                                    - n_common_joints>()
                .setZero();
        }
    };

    template <class SequenceConfig>
    struct SequenceModelSpec<SequenceConfig, model_config::SMPLH> {
        static void set_shape(const Sequence<SequenceConfig>& seq,
                Body<model_config::SMPLH>& body) {
            body.shape().noalias() = seq.shape;
        }
        static void set_pose(const Sequence<SequenceConfig>& seq,
                Body<model_config::SMPLH>& body, size_t frame) {
            constexpr size_t n_common_joints = SequenceConfig::n_body_joints() * 3;
            body.trans().noalias() = seq.trans.row(frame).transpose();
            body.pose().noalias() = seq.pose.row(frame).transpose();
        }
    };
}  // namespace internal

// SEQUENCE interface
// An pose+shape sequence
// SequenceConfig: pick from smplx::sequence_config::*
template<class SequenceConfig>
class Sequence {
public:
    // Create sequence and load from AMASS-like .npz, with fields:
    // trans, gender (optional), mocap_framerate (optional),
    // betas, dmpls, poses.
    // If path is empty, constructs empty sequence (n_frames = 0).
    explicit Sequence(const std::string& path = "");

    // Load sequence from AMASS-like .npz, with fields:
    // trans, gender (optional), mocap_framerate (optional),
    // betas, dmpls, poses
    // Returns true on success
    bool load(const std::string& path);

    // Set body shape
    template<class ModelConfig> inline void set_shape(Body<ModelConfig>& body) {
        internal::SequenceModelSpec<SequenceConfig, ModelConfig>::set_shape(*this, body);
    }
    // Set body pose
    template<class ModelConfig> inline void set_pose(
            Body<ModelConfig>& body, size_t frame) {
        internal::SequenceModelSpec<SequenceConfig, ModelConfig>::set_pose(*this, body, frame);
    }

    // * METADATA
    // Number of frames in sequence
    size_t n_frames;

    // Mocap frame rate
    double frame_rate;
    // Gender, may be unknown
    Gender gender = Gender::unknown;

    // * BODY DATA
    // Extended shape parameters (betas)
    Eigen::Matrix<Scalar, SequenceConfig::n_shape_params(), 1> shape;

    // Root translations
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor> trans;

    // Pose parameters
    Eigen::Matrix<Scalar, Eigen::Dynamic, SequenceConfig::n_pose_params(), Eigen::RowMajor> pose;

    // DMPLs
    Eigen::Matrix<Scalar, Eigen::Dynamic, SequenceConfig::n_dmpls(), Eigen::RowMajor> dmpls;
};

using SequenceAMASS = Sequence<sequence_config::AMASS>;

}  // namespace smplx

#endif  // ifndef SMPLX_SEQUENCE_9512D947_1D9B_478F_AAB5_6E6A846A6828
