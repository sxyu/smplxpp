#pragma once
#ifndef SMPL_HUMAN_3F77A808_CB46_4AF6_A5FD_70CF554F8871
#define SMPL_HUMAN_3F77A808_CB46_4AF6_A5FD_70CF554F8871

#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>

#define _SMPL_MEMBER_ACCESSOR(name, body) inline auto name() {return body;} \
        inline auto name() const {return body;}

namespace smpl {

using Scalar = float;
using MeshIndex = uint32_t;

using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixColMajor = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using PointCloud = Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor>;
using PointCloud2D = Eigen::Matrix<Scalar, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Triangles = Eigen::Matrix<MeshIndex, Eigen::Dynamic, 3, Eigen::RowMajor>;

using SparseMatrixColMajor = Eigen::SparseMatrix<Scalar>;
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;

/** Represents a generic SMPL human model
 *  This defines the pose/shape of an avatar and cannot be manipulated or viewed */
class Model {
public:
    Model() =delete;
    // Construct from .npz at path (standard SMPL-X npz format)
    // path: .npz model path, in data/models/smplx/*.npz
    // uv_path: UV map information path, see data/models/smplx/uv.txt for an example
    // max_n_hand_pca: number of PCA per hand, 6 (=12/2) in SMPL-X paper
    //                   note this is not used when there is no hand in model
    explicit Model(const std::string& path,
                   const std::string& uv_path = "",
                   size_t max_n_hand_pca = 6);

    // Returns true if has UV map
    inline bool has_uv_map() const { return n_uv_verts > 0; }

    // DATA SHAPE INFO

    // # vertices in model
    size_t n_verts;
    // # faces in model
    size_t n_faces;
    // # UV vertices (may be more than n_verts due to seams)
    // 0 if UV not available
    size_t n_uv_verts;

    // # joints, = n_body_joints + n_hand_joints * 2
    size_t n_joints;
    // Number of non-hand (body or face) joints
    size_t n_body_joints;
    // Number of joints per hand
    size_t n_hand_joints;

    // # shape-dep blend shapes, including body and face
    size_t n_shape_blends;

    // # pose-dep blend shapes = 9 * (n_joints - 1)
    size_t n_pose_blends;

    // Number of PCA components for each hand
    size_t n_hand_pca;

    // total # params
    size_t n_params;

    // ** DATA **
    // Kintree: joint parents/children
    Eigen::VectorXi parent;
    std::vector<std::vector<int> > children;

    // Points in the initial mesh, (#verts, 3)
    PointCloud verts;

    // Triangles in the mesh, (#faces, 3)
    Triangles faces;

    // Initial joints
    PointCloud joints;

    // Shape-dependent blend shapes, (3*#joints, #shape blends)
    // each col represents a point cloud (#joints, 3) in row-major order
    MatrixColMajor shape_blend;

    // Pose-dependent blend shapes, (3*#joints, #pose blends)
    // each col represents a point cloud (#joints, 3) in row-major order
    MatrixColMajor pose_blend;

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
    PointCloud2D uv;
    // UV triangles (indices in uv), size (n_faces, 3)
    Triangles uv_triangles;
};

// A particular SMPL instance, with pose/shape/hand parameters.
// Includes parameter vector + cloud data
class Body {
public:
    explicit Body(const Model& model, bool set_zero = true);

    // Perform LBS and output verts
    // enable_pose_blendshapes: if false, disables pose blendshapes;
    //                          this provides a significant speedup at the cost of
    //                          worse accuracy
    void update(bool enable_pose_blendshapes = true);

    // Save as obj file
    void save_obj(const std::string& path) const;

    // Parameter accessors (maps to parts of params)
    // Base position (translation)
    _SMPL_MEMBER_ACCESSOR(trans, data.head<3>());
    // Pose (angle-axis)
    _SMPL_MEMBER_ACCESSOR(pose, data.segment(3, 3 * model.n_body_joints));
    // Hand principal component weights
    _SMPL_MEMBER_ACCESSOR(hand_pca, data.segment(3 + 3 * model.n_body_joints,
                model.n_hand_pca * 2));
    _SMPL_MEMBER_ACCESSOR(hand_pca_l, data.segment(3 + 3 * model.n_body_joints,
                model.n_hand_pca));
    _SMPL_MEMBER_ACCESSOR(hand_pca_r, hand_pca().tail(model.n_body_joints));
    // Shape params
    _SMPL_MEMBER_ACCESSOR(shape, data.tail(model.n_shape_blends));

    void set_zero();

    // The SMPL model used
    const Model& model;

    // * INPUTS
    // Parameters vector
    Vector data;

    // Optional deformation for each vertex, added to base mesh before LBS
    // (empty matrix = no deform, else should be (#verts, 3))
    PointCloud deform;

    // * OUTPUTS generated by update
    // Deformed vertices
    PointCloud verts;

    // Deformed joints
    PointCloud joints;
};

}  // namespace smpl
#endif  // ifndef SMPL_HUMAN_3F77A808_CB46_4AF6_A5FD_70CF554F8871
