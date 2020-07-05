#include "smpl/smpl.hpp"

#include <cnpy.h>
#include <cstring>
#include <algorithm>
#include <fstream>

#include "smpl/util.hpp"

#undef _SMPL_BEGIN_PROFILE
#undef _SMPL_PROFILE
#define _SMPL_BEGIN_PROFILE
#define _SMPL_PROFILE(_)

namespace smpl {
namespace {
// Helpers to allow loading either float/double matrices
template<class KintreeScalar>
inline void load_kintree(const cnpy::NpyArray& kintree_raw, size_t& n_joints,
        Eigen::VectorXi & parent, std::vector<std::vector<int> >& children) {
    // Construct kinematic tree (parent/child rels between joints) from kintree_table
    n_joints = kintree_raw.shape[1];
    parent.resize(n_joints);
    children.resize(n_joints);

    using KintreeMapType = Eigen::Map<const Eigen::Matrix<KintreeScalar, 2, Eigen::Dynamic,
          Eigen::RowMajor>>;
    KintreeMapType kt(kintree_raw.data<KintreeScalar>(), 2, n_joints);
    int max_elem = (int)*std::max_element(kt.data() + kt.cols(), kt.data() + 2 * kt.cols());
    Eigen::VectorXi kintree_map(max_elem + 1);
    kintree_map.setConstant(-1);
    for (size_t i = 0; i < n_joints; ++i) kintree_map[kt(1, i)] = i;
    for (size_t i = 0; i < n_joints; ++i) {
        int idx = (int)kt(0, i);
        if (idx >= 0 && idx < kintree_map.size() && ~kintree_map[idx]) {
            parent[i] = kintree_map[idx];
            children[parent[i]].push_back(i);
        } else {
            parent[i] = -1;
        }
    }
}

// Matrix load helper; currently copies on return
// can modify cnpy to load into the Eigen matrix; not important for now
template<class InputScalar>
inline Matrix load_matrix(const cnpy::NpyArray& raw, size_t r, size_t c) {
    if (raw.fortran_order) {
        return Eigen::template Map<const Eigen::Matrix<InputScalar,
               Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > (raw.data<InputScalar>(), r, c)
                   .template cast<Scalar>();
    } else {
        return Eigen::template Map<const Eigen::Matrix<InputScalar,
               Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > (raw.data<InputScalar>(), r, c)
                   .template cast<Scalar>();
    }
}

void assert_valid_scalar_word_sz(size_t sz) {
    _SMPL_ASSERT(sz == 8 || sz == 4);
}
}  // namespace

Model::Model(const std::string& path, const std::string& uv_path, size_t max_n_hand_pca) {
    _SMPL_BEGIN_PROFILE;
    cnpy::npz_t npz = cnpy::npz_load(path);
    _SMPL_PROFILE(load);

    _SMPL_ASSERT_EQ(npz.count("kintree_table"), 1);
    {
        const cnpy::NpyArray& kintree_raw = npz.at("kintree_table");
        _SMPL_ASSERT_EQ(kintree_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(kintree_raw.shape[0], 2);
        assert_valid_scalar_word_sz(kintree_raw.word_size);

        if (kintree_raw.word_size == 8) {
            load_kintree<int64_t>(kintree_raw, n_joints, parent, children);
        } else {
            load_kintree<int32_t>(kintree_raw, n_joints, parent, children);
        }
    }
    n_pose_blends = 9 * (n_joints - 1);
    _SMPL_PROFILE(kintree);

    _SMPL_ASSERT_EQ(npz.count("v_template"), 1);
    {
        const auto& verts_raw = npz.at("v_template");
        _SMPL_ASSERT_EQ(verts_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(verts_raw.shape[1], 3);
        assert_valid_scalar_word_sz(verts_raw.word_size);
        n_verts = verts_raw.shape[0];
        _SMPL_ASSERT(n_verts > 0);
        verts.resize(n_verts, 3);
        if(verts_raw.word_size == 8) {
            verts.noalias() = load_matrix<double>(verts_raw, n_verts, 3);
        } else {
            verts.noalias() = load_matrix<float>(verts_raw, n_verts, 3);
        }
    }
    _SMPL_PROFILE(verts);

    // Load triangle mesh
    if (npz.count("f")) {
        const auto& faces_raw = npz.at("f");
        _SMPL_ASSERT_EQ(faces_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(faces_raw.shape[1], 3);
        _SMPL_ASSERT(faces_raw.word_size == 4);
        n_faces = faces_raw.shape[0];
        faces.resize(n_faces, 3);
        faces.noalias() = Eigen::template Map<const Triangles>(
                faces_raw.data<MeshIndex>(), n_faces, 3);
        _SMPL_PROFILE(faces);
    } else {
        std::cerr << "WARNING: no triangle faces (key: 'f') in SMPL model npz, "
                     "mesh may not render properly\n";
    }

    // Load joint regressor
    _SMPL_ASSERT_EQ(npz.count("J_regressor"), 1);
    {
        const auto& jreg_raw = npz.at("J_regressor");
        _SMPL_ASSERT_EQ(jreg_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(jreg_raw.shape[0], n_joints);
        _SMPL_ASSERT_EQ(jreg_raw.shape[1], n_verts);
        assert_valid_scalar_word_sz(jreg_raw.word_size);
        joint_reg.resize(n_joints, n_verts);
        if (jreg_raw.word_size == 8) {
            joint_reg = load_matrix<double>(jreg_raw, n_joints, n_verts).sparseView();
        } else {
            joint_reg = load_matrix<float>(jreg_raw, n_joints, n_verts).sparseView();
        }
        joints = joint_reg * verts;
    }
    _SMPL_PROFILE(jreg);

    // Load LBS weights
    _SMPL_ASSERT_EQ(npz.count("weights"), 1);
    {
        const auto& wt_raw = npz.at("weights");
        _SMPL_ASSERT_EQ(wt_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(wt_raw.shape[0], n_verts);
        _SMPL_ASSERT_EQ(wt_raw.shape[1], n_joints);
        assert_valid_scalar_word_sz(wt_raw.word_size);
        weights.resize(n_verts, n_joints);
        if (wt_raw.word_size == 8) {
            weights = load_matrix<double>(wt_raw, n_verts, n_joints).sparseView();
        } else {
            weights = load_matrix<float>(wt_raw, n_verts, n_joints).sparseView();
            Eigen::MatrixXf tmp = load_matrix<float>(wt_raw, n_verts, n_joints);
        }
    }
    _SMPL_PROFILE(weights);

    // Load shape-dep blend shapes
    _SMPL_ASSERT_EQ(npz.count("shapedirs"), 1);
    {
        const auto& sb_raw = npz.at("shapedirs");
        _SMPL_ASSERT_EQ(sb_raw.shape.size(), 3);
        _SMPL_ASSERT_EQ(sb_raw.shape[0], n_verts);
        _SMPL_ASSERT_EQ(sb_raw.shape[1], 3);
        assert_valid_scalar_word_sz(sb_raw.word_size);
        n_shape_blends = sb_raw.shape[2];
        shape_blend.resize(3 * n_verts, n_shape_blends);
        if (sb_raw.word_size == 8) {
            shape_blend.noalias() = load_matrix<double>(sb_raw, 3 * n_verts, n_shape_blends);
        } else {
            shape_blend.noalias() = load_matrix<float>(sb_raw, 3 * n_verts, n_shape_blends);
        }
    }
    _SMPL_PROFILE(shape blends);

    // Load pose-dep blend shapes
    _SMPL_ASSERT_EQ(npz.count("posedirs"), 1);
    {
        const auto& pb_raw = npz.at("posedirs");
        _SMPL_ASSERT_EQ(pb_raw.shape.size(), 3);
        _SMPL_ASSERT_EQ(pb_raw.shape[0], n_verts);
        _SMPL_ASSERT_EQ(pb_raw.shape[1], 3);
        _SMPL_ASSERT_EQ(pb_raw.shape[2], n_pose_blends);
        assert_valid_scalar_word_sz(pb_raw.word_size);
        pose_blend.resize(3 * n_verts, n_pose_blends);
        if (pb_raw.word_size == 8) {
            pose_blend.noalias() = load_matrix<double>(pb_raw, 3 * n_verts, n_pose_blends);
        } else {
            pose_blend.noalias() = load_matrix<float>(pb_raw, 3 * n_verts, n_pose_blends);
        }
    }
    _SMPL_PROFILE(pose blends);

    if (npz.count("hands_meanl") && npz.count("hands_meanr")) {
        // Model has hands (e.g. SMPL-X, SMPL+H), load hand PCA
        const auto& hml_raw = npz.at("hands_meanl");
        const auto& hmr_raw = npz.at("hands_meanr");
        const auto& hcl_raw = npz.at("hands_componentsl");
        const auto& hcr_raw = npz.at("hands_componentsr");

        _SMPL_ASSERT_EQ(hml_raw.shape.size(), 1);
        _SMPL_ASSERT_EQ(hmr_raw.shape.size(), 1);
        _SMPL_ASSERT_EQ(hmr_raw.shape[0], hml_raw.shape[0]);
        assert_valid_scalar_word_sz(hml_raw.word_size);
        assert_valid_scalar_word_sz(hmr_raw.word_size);

        size_t n_hand_params = hml_raw.shape[0];
        _SMPL_ASSERT_EQ(n_hand_params % 3, 0);
        n_hand_joints = n_hand_params / 3;
        n_body_joints = n_joints - n_hand_joints * 2;
        n_hand_pca = std::min(max_n_hand_pca, n_hand_params);

        _SMPL_ASSERT_EQ(hcl_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(hcl_raw.shape[0], hcl_raw.shape[1]);
        _SMPL_ASSERT_EQ(hcl_raw.shape[0], n_hand_params);
        assert_valid_scalar_word_sz(hcl_raw.word_size);

        _SMPL_ASSERT_EQ(hcr_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(hcr_raw.shape[0], hcr_raw.shape[1]);
        _SMPL_ASSERT_EQ(hcr_raw.shape[0], n_hand_params);
        assert_valid_scalar_word_sz(hcr_raw.word_size);

        hand_mean_l.resize(n_hand_params);
        hand_mean_r.resize(n_hand_params);
        if (hml_raw.word_size == 8) {
            hand_mean_l = load_matrix<double>(hml_raw, n_hand_params, 1);
        } else {
            hand_mean_l = load_matrix<float>(hml_raw, n_hand_params, 1);
        }
        if (hmr_raw.word_size == 8) {
            hand_mean_r = load_matrix<double>(hmr_raw, n_hand_params, 1);
        } else {
            hand_mean_r = load_matrix<float>(hmr_raw, n_hand_params, 1);
        }
        hand_comps_l.resize(n_hand_params, n_hand_pca);
        hand_comps_r.resize(n_hand_params, n_hand_pca);
        if (hcl_raw.word_size == 8) {
            hand_comps_l = load_matrix<double>(hcl_raw, n_hand_params, n_hand_params)
                .topRows(n_hand_pca).transpose();
        } else {
            hand_comps_l = load_matrix<float>(hcl_raw, n_hand_params, n_hand_params)
                .topRows(n_hand_pca).transpose();
        }
        if (hcr_raw.word_size == 8) {
            hand_comps_r = load_matrix<double>(hcr_raw, n_hand_params, n_hand_params)
                    .topRows(n_hand_pca).transpose();
        } else {
            hand_comps_r = load_matrix<float>(hcr_raw, n_hand_params, n_hand_params)
                    .topRows(n_hand_pca).transpose();
        }

        _SMPL_PROFILE(hand pca);
    } else {
        // Model has no hands (e.g. SMPL)
        n_body_joints = n_joints;
        n_hand_joints = 0;
        n_hand_pca = 0;
    }

    n_params = 3 +                 // base translatation
               n_body_joints * 3 + // joint rotations
               n_shape_blends +    // shape params
               n_hand_pca * 2;     // hand pca x 2 hands

    //  Maybe load UV
    if (uv_path.size()) {
        std::ifstream ifs(uv_path);
        ifs >> n_uv_verts;
        _SMPL_ASSERT(n_uv_verts > n_verts);
        if (ifs) {
            // Load the uv data
            uv.resize(n_uv_verts, 2);
            for (size_t i = 0; i < n_uv_verts; ++i) {
                ifs >> uv(i, 0) >> uv(i, 1);
            }
            _SMPL_ASSERT((bool)ifs);
            uv_triangles.resize(n_faces, 3);
            for (size_t i = 0; i < n_faces; ++i) {
                _SMPL_ASSERT((bool)ifs);
                ifs >> uv_triangles(i, 0) >> uv_triangles(i, 1) >> uv_triangles(i, 2);
            }
            // uv_triangles.array() -= 1;
        }
    }
}

}  // namespace smpl
