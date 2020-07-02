#pragma once
#ifndef SMPL_HUMAN_3F77A808_CB46_4AF6_A5FD_70CF554F8871
#define SMPL_HUMAN_3F77A808_CB46_4AF6_A5FD_70CF554F8871

#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
namespace smpl {

using Scalar = float;
using TriangleIndex = uint32_t;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using PointCloud = Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor>;

using SparseMatrix = Eigen::SparseMatrix<Scalar>;
using SparseMatrixRowMajor = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
using Triangles = Eigen::Matrix<TriangleIndex, Eigen::Dynamic, 3, Eigen::RowMajor>;

/** Represents a generic SMPL human model
 *  This defines the pose/shape of an avatar and cannot be manipulated or viewed */
class Model {
public:
    Model() =delete;
    // Construct from .npz at path (standard SMPL-X npz format)
    // max_num_hand_pca: number of PCA per hand, 12 (=24/2) in SMPL-X paper
    //                   note this is not used when there is no hand in model
    explicit Model(const std::string& path, size_t max_num_hand_pca = 12);

    // DATA SHAPE INFO

    // # vertices in model
    size_t num_verts;
    // # faces in model
    size_t num_faces;

    // # joints, = num_body_joints + num_hand_joints * 2
    size_t num_joints;
    // Number of non-hand (body or face) joints
    size_t num_body_joints;
    // Number of joints per hand
    size_t num_hand_joints;

    // # shape-dep blend shapes, including somato (body) and face
    size_t num_shape_blends;

    // # pose-dep blend shapes = 9 * (num_joints - 1)
    size_t num_pose_blends;

    // Number of PCA components for each hand
    size_t num_hand_pca;

    // total # params
    size_t num_params;
    // total # params, after PCA applied
    size_t num_params_trans;

    // DATA
    // Kintree: joint parents/children
    Eigen::VectorXi parent;
    std::vector<std::vector<int> > children;

    // Points in the initial mesh, (#verts, 3)
    PointCloud verts;

    // Triangles in the mesh, (#faces, 3)
    Triangles faces;

    // Shape-dependent blend shapes, (#shape blends, 3*#joints)
    // each row represents a point cloud (#joints, 3) in row-major order
    Matrix shape_blend;

    // Pose-dependent blend shapes, (#pose blends, 3*#joints)
    // each row represents a point cloud (#joints, 3) in row-major order
    Matrix pose_blend;

    // Joint regressor: verts -> joints, (#joints, #verts)
    SparseMatrixRowMajor joint_reg;

    // LBS weights, (#verts, #joints)
    SparseMatrix weights;

    // Hand PCA data
    // Hand PCA comps: pca -> joint pos delta
    // 3*#hand joints (=45) * #hand pca
    // columns are PC's
    Matrix hand_comps_l, hand_comps_r;
    // Hand PCA means: mean pos of 3x15 hand joints
    Vector hand_mean_l, hand_mean_r;
};

class Pose {
public:
    Pose() =delete;
    explicit Pose(const Model& model);

    // Perform LBS and update verts
    const PointCloud& update();
    void save_obj(const std::string& path);

    const Model& model;

    Vector params;
    PointCloud verts;
};
}  // namespace smpl
#endif  // ifndef SMPL_HUMAN_3F77A808_CB46_4AF6_A5FD_70CF554F8871
