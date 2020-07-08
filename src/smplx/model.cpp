#include <cnpy.h>

#include <algorithm>
#include <cstring>
#include <fstream>

#include "smplx/smplx.hpp"
#include "smplx/util.hpp"
#include "smplx/version.hpp"

namespace smplx {
namespace {
// Matrix load helper; currently copies on return
// can modify cnpy to load into the Eigen matrix; not important for now
inline Matrix load_float_matrix(const cnpy::NpyArray& raw, size_t r, size_t c) {
    size_t dwidth = raw.word_size;
    _SMPLX_ASSERT(dwidth == 4 || dwidth == 8);
    if (raw.fortran_order) {
        if (dwidth == 4) {
            return Eigen::template Map<const Eigen::Matrix<
                float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                       raw.data<float>(), r, c)
                .template cast<Scalar>();
        } else {
            return Eigen::template Map<const Eigen::Matrix<
                double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                       raw.data<double>(), r, c)
                .template cast<Scalar>();
        }
    } else {
        if (dwidth == 4) {
            return Eigen::template Map<const Eigen::Matrix<
                float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                       raw.data<float>(), r, c)
                .template cast<Scalar>();
        } else {
            return Eigen::template Map<const Eigen::Matrix<
                double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                       raw.data<double>(), r, c)
                .template cast<Scalar>();
        }
    }
}
inline Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
load_uint_matrix(const cnpy::NpyArray& raw, size_t r, size_t c) {
    size_t dwidth = raw.word_size;
    _SMPLX_ASSERT(dwidth == 4 || dwidth == 8);
    if (raw.fortran_order) {
        if (dwidth == 4) {
            return Eigen::template Map<const Eigen::Matrix<
                uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                       raw.data<uint32_t>(), r, c)
                .template cast<Index>();
        } else {
            return Eigen::template Map<const Eigen::Matrix<
                uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                       raw.data<uint64_t>(), r, c)
                .template cast<Index>();
        }
    } else {
        if (dwidth == 4) {
            return Eigen::template Map<const Eigen::Matrix<
                uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                       raw.data<uint32_t>(), r, c)
                .template cast<Index>();
        } else {
            return Eigen::template Map<const Eigen::Matrix<
                uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                       raw.data<uint64_t>(), r, c)
                .template cast<Index>();
        }
    }
}

const size_t ANY_SHAPE = (size_t)-1;
void assert_shape(const cnpy::NpyArray& m,
                  std::initializer_list<size_t> shape) {
    _SMPLX_ASSERT_EQ(m.shape.size(), shape.size());
    size_t idx = 0;
    for (auto& dim : shape) {
        if (dim != ANY_SHAPE)
            _SMPLX_ASSERT_EQ(m.shape[idx], dim);
        ++idx;
    }
}
}  // namespace

template<class ModelConfig>
Model<ModelConfig>::Model(Gender gender)
    : Model(util::find_data_file(std::string(ModelConfig::default_path_prefix) +
                util::gender_to_str(gender) + ".npz"),
            util::find_data_file(ModelConfig::default_uv_path),
            gender)
    { }

template<class ModelConfig>
Model<ModelConfig>::Model(const std::string& path, const std::string& uv_path,
        Gender gender) : gender(gender) {
    if (!std::ifstream(path)) {
        std::cerr << "ERROR: Model '" << path << "' does not exist, "
            "did you download the model following instructions in data/models/README.md?\n";
        std::exit(1);
    }
    cnpy::npz_t npz = cnpy::npz_load(path);

    // Load kintree
    children.resize(n_joints());
    for (size_t i = 1; i < n_joints(); ++i) {
        children[ModelConfig::parent[i]].push_back(i);
    }

    // Load base template
    const auto& verts_raw = npz.at("v_template");
    assert_shape(verts_raw, {n_verts(), 3});
    verts.resize(n_verts(), 3);
    verts.noalias() = load_float_matrix(verts_raw, n_verts(), 3);

    // Load triangle mesh
    const auto& faces_raw = npz.at("f");
    assert_shape(faces_raw, {n_faces(), 3});
    faces.resize(n_faces(), 3);
    faces = load_uint_matrix(faces_raw, n_faces(), 3);

    // Load joint regressor
    const auto& jreg_raw = npz.at("J_regressor");
    assert_shape(jreg_raw, {n_joints(), n_verts()});
    joint_reg.resize(n_joints(), n_verts());
    joint_reg = load_float_matrix(jreg_raw, n_joints(), n_verts()).sparseView();
    joints = joint_reg * verts;
    joint_reg.makeCompressed();

    // Load LBS weights
    const auto& wt_raw = npz.at("weights");
    assert_shape(wt_raw, {n_verts(), n_joints()});
    weights.resize(n_verts(), n_joints());
    weights = load_float_matrix(wt_raw, n_verts(), n_joints()).sparseView();
    weights.makeCompressed();

    blend_shapes.resize(3 * n_verts(), n_blend_shapes());
    // Load shape-dep blend shapes
    const auto& sb_raw = npz.at("shapedirs");
    assert_shape(sb_raw, {n_verts(), 3, n_shape_blends()});
    blend_shapes.template leftCols<n_shape_blends()>().noalias() =
        load_float_matrix(sb_raw, 3 * n_verts(), n_shape_blends());

    // Load pose-dep blend shapes
    const auto& pb_raw = npz.at("posedirs");
    assert_shape(pb_raw, {n_verts(), 3, n_pose_blends()});
    blend_shapes.template rightCols<n_pose_blends()>().noalias() =
        load_float_matrix(pb_raw, 3 * n_verts(), n_pose_blends());

    if (npz.count("hands_meanl") && npz.count("hands_meanr")) {
        // Model has hands (e.g. SMPL-X, SMPL+H), load hand PCA
        const auto& hml_raw = npz.at("hands_meanl");
        const auto& hmr_raw = npz.at("hands_meanr");
        const auto& hcl_raw = npz.at("hands_componentsl");
        const auto& hcr_raw = npz.at("hands_componentsr");

        assert_shape(hml_raw, {ANY_SHAPE});
        assert_shape(hmr_raw, {hml_raw.shape[0]});

        size_t n_hand_params = hml_raw.shape[0];
        _SMPLX_ASSERT_EQ(n_hand_params, n_hand_pca_joints() * 3);

        assert_shape(hcl_raw, {n_hand_params, n_hand_params});
        assert_shape(hcr_raw, {n_hand_params, n_hand_params});

        hand_mean_l.resize(n_hand_params);
        hand_mean_r.resize(n_hand_params);
        hand_mean_l = load_float_matrix(hml_raw, n_hand_params, 1);
        hand_mean_r = load_float_matrix(hmr_raw, n_hand_params, 1);

        hand_comps_l.resize(n_hand_params, n_hand_pca());
        hand_comps_r.resize(n_hand_params, n_hand_pca());
        hand_comps_l = load_float_matrix(hcl_raw, n_hand_params, n_hand_params)
                           .topRows(n_hand_pca()).transpose();
        hand_comps_r = load_float_matrix(hcr_raw, n_hand_params, n_hand_params)
                           .topRows(n_hand_pca()).transpose();
    }

    // Maybe load UV (UV mapping WIP)
    if (uv_path.size()) {
        std::ifstream ifs(uv_path);
        ifs >> n_uv_verts;
        if (n_uv_verts) {
            // Currently we only support cases where there areat least as many uv
            // verts as vertices
            _SMPLX_ASSERT(n_uv_verts >= n_verts());
            if (ifs) {
                // Load the uv data
                uv.resize(n_uv_verts, 2);
                for (size_t i = 0; i < n_uv_verts; ++i) ifs >> uv(i, 0) >> uv(i, 1);
                _SMPLX_ASSERT((bool)ifs);
                uv_triangles.resize(n_faces(), 3);
                for (size_t i = 0; i < n_faces(); ++i) {
                    _SMPLX_ASSERT((bool)ifs);
                    ifs >> uv_triangles(i, 0) >> uv_triangles(i, 1) >>
                        uv_triangles(i, 2);
                }
                uv_triangles.array() -= 1;
            }
        }
    }
#ifdef SMPLX_CUDA_ENABLED
    _cuda_load();
#endif
}

template<class ModelConfig>
Model<ModelConfig>::~Model() {
#ifdef SMPLX_CUDA_ENABLED
    _cuda_free();
#endif
}


// Instantiations
template class Model<model_config::SMPL>;
template class Model<model_config::SMPLH>;
template class Model<model_config::SMPLX>;

// Model config constexpr arrays
namespace model_config {
constexpr size_t SMPLX::parent[];
constexpr size_t SMPLH::parent[];
constexpr size_t SMPL::parent[];
constexpr const char* SMPLX::joint_name[];
constexpr const char* SMPLH::joint_name[];
constexpr const char* SMPL::joint_name[];
}  // namespace model_config

}  // namespace smplx
