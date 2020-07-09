#pragma once
#ifndef SMPLX_SMPLX_3F77A808_CB46_4AF6_A5FD_70CF554F8871
#define SMPLX_SMPLX_3F77A808_CB46_4AF6_A5FD_70CF554F8871

#include "smplx/defs.hpp"
#include "smplx/model_config.hpp"

#include <string>
#include <vector>

#define __SMPLX_MEMBER_ACCESSOR(name, body) inline auto name() {return body;} \
        inline auto name() const {return body;}

namespace smplx {
#ifdef SMPLX_CUDA_ENABLED
namespace internal {
// Basic CSR sparse matrix repr
struct GPUSparseMatrix {
    float* values = nullptr;
    int* inner = nullptr;
    int* outer = nullptr;
    int rows, cols, nnz;
};
}  // namespace internal
#endif

/** Represents a generic SMPL human model
 *  This defines the pose/shape of an avatar and cannot be manipulated or viewed
 *  ModelConfig: static 'model configuration', pick from smplx::model_config::SMPL/SMPLH/SMPLX*/
template<class ModelConfig>
class Model {
public:
    // Construct from .npz at default path for given gender
    explicit Model(Gender gender = Gender::neutral);

    // Construct from .npz at path (standard SMPL-X npz format)
    // path: .npz model path, in data/models/smplx/*.npz
    // uv_path: UV map information path, see data/models/smplx/uv.txt for an example
    // gender: records gender of model. For informational purposes only.
    explicit Model(const std::string& path,
                   const std::string& uv_path = "",
                   Gender gender = Gender::unknown);
    ~Model();

    // Load from .npz at default path for given gender
    // useful for dynamically switching genders
    void load(Gender gender = Gender::neutral);
    // Load from .npz at path (standard SMPL-X npz format)
    // path: .npz model path, in data/models/smplx/*.npz
    // uv_path: UV map information path, see data/models/smplx/uv.txt for an example
    // gender: records gender of model. For informational purposes only.
    void load(const std::string& path,
                   const std::string& uv_path = "",
                   Gender new_gender = Gender::unknown);

    Model& operator=(const Model& other) =delete;
    Model& operator=(Model&& other) =delete;

    // Returns true if has UV map
    inline bool has_uv_map() const { return n_uv_verts > 0; }

    using Config = ModelConfig;

    // DATA SHAPE INFO (shorthand) from ModelConfig

    // Number of vertices in model
    static constexpr size_t n_verts() { return Config::n_verts(); }
    // Number of faces in model
    static constexpr size_t n_faces() { return Config::n_faces(); }
    // Number UV vertices (may be more than n_verts due to seams)
    // 0 if UV not available
    // NOTE: not a constexpr
    size_t n_uv_verts;

    // Total number of joints = n_explicit_joints + n_hand_pca_joints * 2
    static constexpr size_t n_joints() { return Config::n_joints(); }
    // Number of explicit joint parameters stored as angle-axis
    static constexpr size_t n_explicit_joints() { return Config::n_explicit_joints(); }
    // Number of joints per hand implicit computed from PCA
    static constexpr size_t n_hand_pca_joints() { return Config::n_hand_pca_joints(); }

    // Total number of blend shapes = n_shape_blends + n_pose_blends
    static constexpr size_t n_blend_shapes() { return Config::n_blend_shapes(); }
    // Number of shape-dep blend shapes, including body and face
    static constexpr size_t n_shape_blends() { return Config::n_shape_blends(); }
    // Number of pose-dep blend shapes = 9 * (n_joints - 1)
    static constexpr size_t n_pose_blends() { return Config::n_pose_blends(); }

    // Number of PCA components for each hand
    static constexpr size_t n_hand_pca() { return Config::n_hand_pca(); }

    // Total number of params = 3 + 3 * n_body_joints + 2 * n_hand_pca +
    static constexpr size_t n_params() { return Config::n_params(); }

    // Model name
    static constexpr const char* name() { return Config::model_name; }
    // Joint names name
    static constexpr const char* joint_name(size_t joint) { return Config::joint_name[joint]; }
    // Parent joint
    static constexpr size_t parent(size_t joint) { return Config::parent[joint]; }

    // Model gender, may be unknown
    Gender gender;

    // ** DATA **
    // Kinematic tree: joint children
    std::vector<std::vector<int> > children;

    // Points in the initial mesh, (#verts, 3)
    Points verts;

    // Triangles in the mesh, (#faces, 3)
    Triangles faces;

    // Initial joint positions
    Points joints;

    // Shape-dependent blend shapes, (3*#joints, #shape blends + #pose blends)
    // each col represents a point cloud (#joints, 3) in row-major order
    Eigen::Matrix<Scalar, Eigen::Dynamic, n_blend_shapes()> blend_shapes;

    // Joint regressor: verts -> joints, (#joints, #verts)
    SparseMatrix joint_reg;

    // LBS weights, (#verts, #joints)
    SparseMatrixColMajor weights;

    // ** Hand PCA data **
    // Hand PCA comps: pca -> joint pos delta
    // 3*#hand joints (=45) * #hand pca
    // columns are PC's
    Matrix hand_comps_l, hand_comps_r;
    // Hand PCA means: mean pos of 3x15 hand joints
    Vector hand_mean_l, hand_mean_r;

    // ** UV Data **, available if has_uv_map()
    // UV coordinates, size (n_uv_verts, 2)
    Points2D uv;
    // UV triangles (indices in uv), size (n_faces, 3)
    Triangles uv_triangles;

#ifdef SMPLX_CUDA_ENABLED
    // ADVANCED: GPU data pointers
    struct {
        float* verts = nullptr;
        float* blend_shapes = nullptr;
        internal::GPUSparseMatrix joint_reg;
        float* joint_reg_dense;
        internal::GPUSparseMatrix weights;
        float* hand_comps_l = nullptr, * hand_comps_r = nullptr;
        float* hand_mean_l = nullptr, * hand_mean_r = nullptr;
    } device;
private:
    void _cuda_load();
    void _cuda_free();
#endif
};
// SMPL Model
using ModelS = Model<model_config::SMPL>;
// SMPL+H Model
using ModelH = Model<model_config::SMPLH>;
// SMPL-X Model with hand joint rotations
using ModelX = Model<model_config::SMPLX>;
// SMPL-X Model with hand PCA
using ModelXpca = Model<model_config::SMPLXpca>;

// A particular SMPL instance, with pose/shape/hand parameters.
// Includes parameter vector + cloud data
template<class ModelConfig>
class Body {
public:
    // Construct body from model
    // set_zero: set to false to leave parameter array uninitialized
    explicit Body(const Model<ModelConfig>& model, bool set_zero = true);
    ~Body();

    // Perform LBS and output verts
    // enable_pose_blendshapes: if false, disables pose blendshapes;
    //                          this provides a significant speedup at the cost of
    //                          worse accuracy
    void update(bool force_cpu = false, bool enable_pose_blendshapes = true);

    // Save as obj file
    void save_obj(const std::string& path) const;

    using Config = ModelConfig;

    // Parameter accessors (maps to parts of params)
    // Base position (translation)
    __SMPLX_MEMBER_ACCESSOR(trans, params.template head<3>());
    // Pose (angle-axis)
    __SMPLX_MEMBER_ACCESSOR(pose, params.template segment<ModelConfig::n_explicit_joints() * 3>(3));
    // Hand principal component weights
    __SMPLX_MEMBER_ACCESSOR(hand_pca, params.segment<ModelConfig::n_hand_pca() * 2>(3 + 3 * model.n_explicit_joints()));
    __SMPLX_MEMBER_ACCESSOR(hand_pca_l, params.template segment<ModelConfig::n_hand_pca()>(3 + 3 * model.n_explicit_joints()));
    __SMPLX_MEMBER_ACCESSOR(hand_pca_r, params.template segment<ModelConfig::n_hand_pca()>(3 + 3 * model.n_explicit_joints() + model.n_hand_pca()));
    // Shape params
    __SMPLX_MEMBER_ACCESSOR(shape, params.template tail<ModelConfig::n_shape_blends()>());

    // * OUTPUTS accessors
    // Get deformed body vertices, in same order as model.verts;
    // must call update() before this is available
    const Points& verts() const;

    // Get deformed body joints, in same order as model.joints;
    // must call update() before this is available
    const Points& joints() const;

    // Set parameters to zero
    inline void set_zero() { params.setZero(); }

    // Set parameters uar in [-0.25, 0.25]
    inline void set_random() { params.setRandom() * 0.25; }

    // The SMPL model used
    const Model<ModelConfig>& model;

    // * INPUTS
    // Parameters vector
    Vector params;

private:
    // * OUTPUTS generated by update
    // Deformed vertices (only shape applied); not available in case of GPU
    // (only device.verts_shaped)
    Points _verts_shaped;

    // Deformed vertices (shape and pose applied)
    mutable Points _verts;

    // Deformed joints (only shape applied)
    Points _joints_shaped;

    // Homogeneous transforms at each joint (bottom row omitted)
    Eigen::Matrix<Scalar, Eigen::Dynamic, 12, Eigen::RowMajor> _joint_transforms;

    // Deformed joints (shape and pose applied)
    mutable Points _joints;

#ifdef SMPLX_CUDA_ENABLED
public:
    struct {
        float* verts = nullptr;
        float* verts_shaped = nullptr;
        float* joints_shaped = nullptr;
        // Internal temp
        float* verts_tmp = nullptr;
        // Internal (#total blend shapes) rm
        float* blendshape_params = nullptr;
        // Internal (#joints, 12) rm
        float* joint_transforms = nullptr;
    } device;

private:
    mutable bool verts_retrieved, joints_retrieved;
    bool _last_update_used_gpu;

    void _cuda_load();
    void _cuda_free();
    void _cuda_maybe_retrieve_verts() const;
    void _cuda_update(
            float * h_blendshape_params,
            float * h_joint_transforms,
            bool enable_pose_blendshapes = true);
#endif
};
// SMPL Body
using BodyS = Body<model_config::SMPL>;
// SMPL-H Body
using BodyH = Body<model_config::SMPLH>;
// SMPL-X Body with hand joint rotations
using BodyX = Body<model_config::SMPLX>;
// SMPL-X Body with hand PCA
using BodyXpca = Body<model_config::SMPLXpca>;

}  // namespace smpl

#endif  // ifndef SMPLX_SMPLX_3F77A808_CB46_4AF6_A5FD_70CF554F8871
