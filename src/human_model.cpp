#include "human.hpp"

#include <cnpy.h>
#include <cstring>
#include <algorithm>
#include <map>

#include "util.hpp"

namespace smpl {
namespace {
    template<class KintreeScalar>
    inline void load_kintree(const cnpy::NpyArray& kintree_raw,
            size_t& num_joints,
            Eigen::VectorXi & parent, std::vector<std::vector<int> >& children) {
        num_joints = kintree_raw.shape[1];
        parent.resize(num_joints);
        children.resize(num_joints);

        using KintreeMapType = Eigen::Map<const Eigen::Matrix<KintreeScalar, 2, Eigen::Dynamic,
              Eigen::RowMajor>>;
        KintreeMapType kt(kintree_raw.data<KintreeScalar>(), 2, num_joints);
        int max_elem = (int)*std::max_element(kt.data() + kt.cols(), kt.data() + 2 * kt.cols());
        Eigen::VectorXi kintree_map(max_elem + 1);
        kintree_map.setConstant(-1);
        for (size_t i = 0; i < num_joints; ++i) kintree_map[kt(1, i)] = i;
        for (size_t i = 0; i < num_joints; ++i) {
            int idx = (int)kt(0, i);
            if (idx >= 0 && idx < kintree_map.size() && ~kintree_map[idx]) {
                parent[i] = kintree_map[idx];
                children[parent[i]].push_back(i);
            } else {
                parent[i] = -1;
            }
        }
    }

    template<class InputScalar>
    inline Matrix load_matrix(const cnpy::NpyArray& raw, size_t r, size_t c) {
        return Eigen::template Map<const Eigen::Matrix<InputScalar,
               Eigen::Dynamic, Eigen::Dynamic> > (raw.data<InputScalar>(), r, c)
                   .template cast<Scalar>();
    }

    void assert_valid_scalar_word_sz(size_t sz) {
        _SMPL_ASSERT(sz == 8 || sz == 4);
    }
}  // namespace

Model::Model(const std::string& path, size_t max_num_hand_pca) {
    _SMPL_BEGIN_PROFILE;
    cnpy::npz_t npz = cnpy::npz_load(path);
    _SMPL_PROFILE(load);

    {
        const cnpy::NpyArray& kintree_raw = npz.at("kintree_table");
        // verts.resize(verts_raw.shape[0], verts_raw.shape[1]);
        // std::memcpy(verts.data(), &(*verts_raw.data_holder)[0],
        //             verts_raw.num_bytes());
        _SMPL_ASSERT_EQ(kintree_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(kintree_raw.shape[0], 2);
        assert_valid_scalar_word_sz(kintree_raw.word_size);

        if (kintree_raw.word_size == 8) {
            load_kintree<int64_t>(kintree_raw, num_joints, parent, children);
        } else {
            load_kintree<int32_t>(kintree_raw, num_joints, parent, children);
        }
    }
    num_pose_blends = 9 * (num_joints - 1);
    _SMPL_PROFILE(kintree);

    {
        const auto& verts_raw = npz.at("v_template");
        _SMPL_ASSERT_EQ(verts_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(verts_raw.shape[1], 3);
        assert_valid_scalar_word_sz(verts_raw.word_size);
        num_verts = verts_raw.shape[0];
        verts.resize(num_verts, 3);
        if(verts_raw.word_size == 8) {
            verts.noalias() = load_matrix<double>(verts_raw, num_verts, 3);
        } else {
            verts.noalias() = load_matrix<float>(verts_raw, num_verts, 3);
        }
    }
    _SMPL_PROFILE(verts);

    {
        const auto& faces_raw = npz.at("f");
        _SMPL_ASSERT_EQ(faces_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(faces_raw.shape[1], 3);
        _SMPL_ASSERT(faces_raw.word_size == 4);
        num_faces = faces_raw.shape[0];
        faces.resize(num_faces, 3);
        faces.noalias() = Eigen::template Map<const Triangles>(
                faces_raw.data<TriangleIndex>(), num_faces, 3);
    }
    _SMPL_PROFILE(faces);

    // Load joint regressor
    {
        const auto& jreg_raw = npz.at("J_regressor");
        _SMPL_ASSERT_EQ(jreg_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(jreg_raw.shape[0], num_joints);
        _SMPL_ASSERT_EQ(jreg_raw.shape[1], num_verts);
        assert_valid_scalar_word_sz(jreg_raw.word_size);
        joint_reg.resize(num_joints, num_verts);
        if (jreg_raw.word_size == 8) {
            joint_reg = load_matrix<double>(jreg_raw, num_joints, num_verts).sparseView();
        } else {
            joint_reg = load_matrix<float>(jreg_raw, num_joints, num_verts).sparseView();
        }
    }
    _SMPL_PROFILE(jreg);

    // Load LBS weights
    {
        const auto& wt_raw = npz.at("weights");
        _SMPL_ASSERT_EQ(wt_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(wt_raw.shape[0], num_verts);
        _SMPL_ASSERT_EQ(wt_raw.shape[1], num_joints);
        assert_valid_scalar_word_sz(wt_raw.word_size);
        weights.resize(num_verts, num_joints);
        if (wt_raw.word_size == 8) {
            weights = load_matrix<double>(wt_raw, num_verts, num_joints).sparseView();
        } else {
            weights = load_matrix<float>(wt_raw, num_verts, num_joints).sparseView();
        }
    }
    _SMPL_PROFILE(weights);

    // Load shape-dep blend shapes
    {
        const auto& sb_raw = npz.at("shapedirs");
        _SMPL_ASSERT_EQ(sb_raw.shape.size(), 3);
        _SMPL_ASSERT_EQ(sb_raw.shape[0], num_verts);
        _SMPL_ASSERT_EQ(sb_raw.shape[1], 3);
        assert_valid_scalar_word_sz(sb_raw.word_size);
        num_shape_blends = sb_raw.shape[2];
        shape_blend.resize(3 * num_verts, num_shape_blends);
        if (sb_raw.word_size == 8) {
            shape_blend.noalias() =
                load_matrix<double>(sb_raw, 3 * num_verts, num_shape_blends).transpose();
        } else {
            shape_blend.noalias() =
                load_matrix<float>(sb_raw, 3 * num_verts, num_shape_blends).transpose();
        }
    }
    _SMPL_PROFILE(shape blends);

    // Load pose-dep blend shapes
    {
        const auto& pb_raw = npz.at("posedirs");
        _SMPL_ASSERT_EQ(pb_raw.shape.size(), 3);
        _SMPL_ASSERT_EQ(pb_raw.shape[0], num_verts);
        _SMPL_ASSERT_EQ(pb_raw.shape[1], 3);
        _SMPL_ASSERT_EQ(pb_raw.shape[2], num_pose_blends);
        assert_valid_scalar_word_sz(pb_raw.word_size);
        pose_blend.resize(num_pose_blends, 3 * num_verts);
        if (pb_raw.word_size == 8) {
            pose_blend.noalias() = load_matrix<double>(pb_raw, 3 * num_verts, num_pose_blends)
                .transpose();
        } else {
            pose_blend.noalias() = load_matrix<float>(pb_raw, 3 * num_verts, num_pose_blends)
                .transpose();
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

        num_hand_joints = hml_raw.shape[0];
        num_body_joints = num_joints - num_hand_joints * 2;
        num_hand_pca = std::min(max_num_hand_pca, num_hand_joints);

        _SMPL_ASSERT_EQ(hcl_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(hcl_raw.shape[0], hcl_raw.shape[1]);
        _SMPL_ASSERT_EQ(hcl_raw.shape[0], num_hand_joints);
        assert_valid_scalar_word_sz(hcl_raw.word_size);

        _SMPL_ASSERT_EQ(hcr_raw.shape.size(), 2);
        _SMPL_ASSERT_EQ(hcr_raw.shape[0], hcr_raw.shape[1]);
        _SMPL_ASSERT_EQ(hcr_raw.shape[0], num_hand_joints);
        assert_valid_scalar_word_sz(hcr_raw.word_size);

        hand_mean_l.resize(num_hand_joints);
        hand_mean_r.resize(num_hand_joints);
        if (hml_raw.word_size == 8) {
            hand_mean_l = load_matrix<double>(hml_raw, num_hand_joints, 1);
        } else {
            hand_mean_l = load_matrix<float>(hml_raw, num_hand_joints, 1);
        }
        if (hmr_raw.word_size == 8) {
            hand_mean_r = load_matrix<double>(hmr_raw, num_hand_joints, 1);
        } else {
            hand_mean_r = load_matrix<float>(hmr_raw, num_hand_joints, 1);
        }

        hand_comps_l.resize(num_hand_joints, num_hand_pca);
        hand_comps_r.resize(num_hand_joints, num_hand_pca);
        if (hcl_raw.word_size == 8) {
            hand_comps_l = load_matrix<double>(hcl_raw, num_hand_joints, num_hand_joints)
                .topRows(num_hand_pca).transpose();
        } else {
            hand_comps_l = load_matrix<float>(hcl_raw, num_hand_joints, num_hand_joints)
                .topRows(num_hand_pca).transpose();
        }
        if (hcr_raw.word_size == 8) {
            hand_comps_r = load_matrix<double>(hcr_raw, num_hand_joints, num_hand_joints)
                    .topRows(num_hand_pca).transpose();
        } else {
            hand_comps_r = load_matrix<float>(hcr_raw, num_hand_joints, num_hand_joints)
                    .topRows(num_hand_pca).transpose();
        }

        _SMPL_PROFILE(hand pca);
    } else {
        // Model has no hands (e.g. SMPL)
        num_body_joints = num_joints;
        num_hand_joints = 0;
        num_hand_pca = 0;
    }
}

}  // namespace smpl
