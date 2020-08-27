#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <string>

#include <smplx/smplx.hpp>
#include <smplx/sequence.hpp>
#include <smplx/util.hpp>

namespace py = pybind11;
using namespace smplx;

namespace {
const bool CUDA_AVAILABLE =
#ifdef SMPLX_CUDA_ENABLED
    true
#else
    false
#endif
    ;

template <class ModelConfig>
void declare_model(py::module& m, const std::string& py_model_name,
                   const std::string& py_body_name) {
    using ModelClass = Model<ModelConfig>;
    using BodyClass = Body<ModelConfig>;
    py::class_<ModelClass>(m, py_model_name.c_str())
        .def(py::init<Gender>(), py::arg("gender") = Gender::neutral)
        .def(py::init<const std::string&, const std::string&, Gender>(),
             py::arg("path"), py::arg("uv_path") = "",
             py::arg("gender") = Gender::unknown)
        .def("load", py::overload_cast<Gender>(&ModelClass::load),
             "Load default npz model for given gender", py::arg("gender"))
        .def("load",
             py::overload_cast<const std::string&, const std::string&, Gender>(
                 &ModelClass::load),
             "Load npz model", py::arg("path"), py::arg("uv_path") = "",
             py::arg("gender") = Gender::unknown)
        .def("set_deformations", &ModelClass::set_deformations,
             "Set template deformations: verts := verts_init + deform",
             py::arg("deform") = Gender::unknown)
        .def("set_template", &ModelClass::set_template,
             "Set base template: verts := template")
        .def_property_readonly_static(
            "n_verts",
            [](const py::object& obj) { return ModelClass::n_verts(); },
            "Number of vertices")
        .def_property_readonly_static(
            "n_joints",
            [](const py::object& obj) { return ModelClass::n_joints(); },
            "Number of joints")
        .def_property_readonly_static(
            "n_faces",
            [](const py::object& obj) { return ModelClass::n_faces(); },
            "Number of faces in mesh")
        .def_property_readonly_static(
            "n_blend_shapes",
            [](const py::object& obj) { return ModelClass::n_blend_shapes(); },
            "Total number of blend shapes")
        .def_property_readonly_static(
            "n_pose_blends",
            [](const py::object& obj) { return ModelClass::n_pose_blends(); },
            "Number of pose blend shapes (9 * n_joints)")
        .def_property_readonly_static(
            "n_shape_blends",
            [](const py::object& obj) { return ModelClass::n_shape_blends(); },
            "Number of shape blend shapes")
        .def_property_readonly_static(
            "n_hand_pca",
            [](const py::object& obj) { return ModelClass::n_hand_pca(); },
            "Number of hand PCA components (only if model has hand PCA)")
        .def_property_readonly_static(
            "n_explicit_joints",
            [](const py::object& obj) {
                return ModelClass::n_explicit_joints();
            },
            "Number of explicit joints (joints controlled using rotations "
            "rather than PCA)")
        .def_property_readonly_static(
            "n_hand_pca_joints",
            [](const py::object& obj) {
                return ModelClass::n_hand_pca_joints();
            },
            "Number of hand pca joints (joints controlled using hand PCA "
            "rather than using rotations)")
        .def_property_readonly_static(
            "n_params",
            [](const py::object& obj) { return ModelClass::n_params(); },
            "Total number of model parameters")
        .def_property_readonly_static(
            "name", [](const py::object& obj) { return ModelClass::name(); },
            "Model name")
        .def_static("joint_name", &ModelClass::joint_name,
                    "Name of joint at given index")
        .def_static("parent", &ModelClass::parent, "Index of parent joint")
        .def_readonly("gender", &ModelClass::gender, "Gender (may be unknown)")
        .def_property_readonly(
            "n_uv_verts", &ModelClass::n_uv_verts,
            "Total number of texture coordinates (UV vertices)")
        .def_property_readonly("has_uv_map", &ModelClass::has_uv_map,
                               "Returns true if UV map is available")

        .def_readonly("children", &ModelClass::children,
                      "Kinematic tree children indices")
        .def_readonly("verts", &ModelClass::verts, "Unposed vertices")
        .def_readonly("vertices", &ModelClass::verts,
                      "Unposed vertices (alias)")
        .def_readonly("joints", &ModelClass::joints, "Unposed joints")
        .def_readonly("faces", &ModelClass::faces, "Triangular faces")
        .def_readonly("joint_reg", &ModelClass::joint_reg,
                      "Joint regressor sparsematrix (n_joints, n_verts)")
        .def_readonly("weights", &ModelClass::weights,
                      "LBS weights sparse matrix (n_verts, n_joints)")
        .def_readonly("blend_shapes", &ModelClass::blend_shapes,
                      "Shape and pose blend shapes "
                      "(3 * n_verts, n_shape_blends + n_pose_blends) colmajor;"
                      "each column is (n_verts, 3) rowmajor")
        .def_property_readonly(
            "has_hand_pca",
            [](const py::object& _) { return ModelConfig::n_hand_pca() > 0; },
            "True if hand PCA is available")
        .def_readonly("hand_comps_l", &ModelClass::hand_comps_l,
                      "Principal components for left hand (3 * "
                      "n_hand_pca_joints, n_hand_pca). "
                      "Available if has_hand_pca")
        .def_readonly("hand_comps_r", &ModelClass::hand_comps_r,
                      "Principal components for right hand (3 * "
                      "n_hand_pca_joints, n_hand_pca). "
                      "Available if has_hand_pca")
        .def_readonly("hand_mean_l", &ModelClass::hand_mean_l,
                      "Mean parameters for left hand (3 * "
                      "n_hand_pca_joints). "
                      "Available if has_hand_pca")
        .def_readonly("hand_mean_r", &ModelClass::hand_mean_r,
                      "Mean parameters for right hand (3 * "
                      "n_hand_pca_joints). "
                      "Available if has_hand_pca")
        .def_readonly("uv", &ModelClass::uv,
                      "Texture coords (n_uv_verts, 2). "
                      "Available if has_uv_map")
        .def_readonly(
            "uv_faces", &ModelClass::uv_faces,
            "Texture coord faces (n_faces, 3). Available if has_uv_map")
        .def("__repr__", [](const ModelClass& obj) {
            return std::string("<smplxpp.Model(name=") + obj.name() +
                   ", gender=" + util::gender_to_str(obj.gender) +
                   ", n_params=" + std::to_string(obj.n_params()) +
                   ", n_verts=" + std::to_string(obj.n_verts()) +
                   ", n_joints=" + std::to_string(obj.n_joints()) +
                   ", n_faces=" + std::to_string(obj.n_faces()) +
                   ", n_shape_blends=" + std::to_string(obj.n_shape_blends()) +
                   ", has_uv=" + (obj.has_uv_map() ? "True" : "False") + ")>";
        });
    using TransRefType = Eigen::Ref<Eigen::Matrix<Scalar, 3, 1>>;
    using TransConstRefType = Eigen::Ref<const Eigen::Matrix<Scalar, 3, 1>>;
    using PoseRefType = Eigen::Ref<
        Eigen::Matrix<Scalar, ModelConfig::n_explicit_joints() * 3, 1>>;
    using PoseConstRefType = Eigen::Ref<
        const Eigen::Matrix<Scalar, ModelConfig::n_explicit_joints() * 3, 1>>;
    using HandPCARefType =
        Eigen::Ref<Eigen::Matrix<Scalar, ModelConfig::n_hand_pca() * 2, 1>>;
    using HandPCAConstRefType = Eigen::Ref<
        const Eigen::Matrix<Scalar, ModelConfig::n_hand_pca() * 2, 1>>;
    using HandPCAHalfRefType =
        Eigen::Ref<Eigen::Matrix<Scalar, ModelConfig::n_hand_pca(), 1>>;
    using HandPCAHalfConstRefType =
        Eigen::Ref<const Eigen::Matrix<Scalar, ModelConfig::n_hand_pca(), 1>>;
    using ShapeRefType =
        Eigen::Ref<Eigen::Matrix<Scalar, ModelConfig::n_shape_blends(), 1>>;
    using ShapeConstRefType = Eigen::Ref<
        const Eigen::Matrix<Scalar, ModelConfig::n_shape_blends(), 1>>;

    py::class_<BodyClass>(m, py_body_name.c_str())
        .def(py::init<const ModelClass&, bool>(), py::arg("model"),
             py::arg("set_zero") = true)
        .def("update", &BodyClass::update, py::arg("force_cpu") = false,
             py::arg("enable_pose_blendshapes") = true)
        .def_property_readonly("verts", &BodyClass::verts,
                               "Posed vertices, available after update() call")
        .def_property_readonly(
            "vertices", &BodyClass::verts,
            "Posed vertices, available after update() call (alias)")
        .def_property_readonly("joints", &BodyClass::joints,
                               "Posed joints, available after update() call")
        .def_property_readonly("joint_transforms", &BodyClass::joint_transforms,
                               "Joint transforms. Each row is a row-major "
                               "(3,4) rigid body transform matrix, bottom row "
                               "omitted. Available after update() call")
        .def_property_readonly(
            "vert_transforms", &BodyClass::vert_transforms,
            "Vertex transforms. Each row is a row-major (3,4) "
            "rigid body transform matrix, bottom row "
            "omitted. Available after update() call")
        .def_property_readonly(
            "model",
            [](const BodyClass& obj) -> const ModelClass& { return obj.model; },
            "The associated model instance")
        .def_readwrite("params", &BodyClass::params, "Parameters vector")
        .def_property(
            "trans", [](BodyClass& obj) -> TransRefType { return obj.trans(); },
            [](BodyClass& obj, const TransConstRefType& val) {
                obj.trans() = val;
            },
            "Translation part of parameters vector (3)")
        .def_property(
            "pose", [](BodyClass& obj) -> PoseRefType { return obj.pose(); },
            [](BodyClass& obj, const PoseConstRefType& val) {
                obj.pose() = val;
            },
            "Pose part of parameters vector (3 * n_explicit_joints) in "
            "axis-angle")
        .def_property(
            "hand_pca",
            [](BodyClass& obj) -> HandPCARefType { return obj.hand_pca(); },
            [](BodyClass& obj, const HandPCAConstRefType& val) {
                obj.hand_pca() = val;
            },
            "Hand PCA part of parameters vector, for both hands (2 * "
            "n_hand_pca)")
        .def_property(
            "hand_pca_l",
            [](BodyClass& obj) -> HandPCAHalfRefType {
                return obj.hand_pca_l();
            },
            [](BodyClass& obj, const HandPCAHalfConstRefType& val) {
                obj.hand_pca_l() = val;
            },
            "Hand PCA part of parameters vector, left hand "
            "(n_hand_pca)")
        .def_property(
            "hand_pca_r",
            [](BodyClass& obj) -> HandPCAHalfRefType {
                return obj.hand_pca_r();
            },
            [](BodyClass& obj, const HandPCAHalfConstRefType& val) {
                obj.hand_pca_r() = val;
            },
            "Hand PCA part of parameters vector, right hand "
            "(n_hand_pca)")
        .def_property(
            "shape", [](BodyClass& obj) -> ShapeRefType { return obj.shape(); },
            [](BodyClass& obj, const ShapeConstRefType& val) {
                obj.shape() = val;
            },
            "Shape part of parameters vector (n_shape_blends)")
        .def("set_zero", &BodyClass::set_zero, "Set all parameters to 0")
        .def("set_random", &BodyClass::set_random,
             "Set all parameters u.a.r. in  [-0.25, 0.25]. Maybe not "
             "the "
             "best "
             "distribution.")
        .def("save_obj", &BodyClass::save_obj,
             "Save a basic OBJ file from the posed model (call update first)")
        .def("__repr__", [](const BodyClass& obj) {
            return std::string("<smplxpp.Body(name=") + obj.model.name() +
                   ", gender=" + util::gender_to_str(obj.model.gender) +
                   ", n_params=" + std::to_string(obj.model.n_params()) +
                   ", n_verts=" + std::to_string(obj.model.n_verts()) +
                   ", n_joints=" + std::to_string(obj.model.n_joints()) + ")>";
        });
}

template <class SequenceConfig, class ModelConfig>
void declare_sequence_model_spec(py::class_<Sequence<SequenceConfig>>& cl) {
    using SeqClass = Sequence<SequenceConfig>;
    cl.def("set_pose", &SeqClass::template set_pose<ModelConfig>,
           "Set body's pose+translation parameters, given a frame number in "
           "the sequence");
    cl.def("set_shape", &SeqClass::template set_shape<ModelConfig>,
           "Set body's shape parameters");
}

template <class SequenceConfig>
void declare_sequence(py::module& m, const std::string& py_sequence_name) {
    using SeqClass = Sequence<SequenceConfig>;
    auto cl =
        py::class_<SeqClass>(m, py_sequence_name.c_str())
            .def(py::init<const std::string&>(), py::arg("amass_npz_path") = "")
            .def("load", &SeqClass::load, py::arg("amass_npz_path"))

            .def_property_readonly_static(
                "n_pose_params",
                [](const py::object& _) {
                    return SequenceConfig::n_pose_params();
                },
                "Size of pose space in dataset = (n_body_joints + 2 * "
                "n_hand_joints) * 3")
            .def_property_readonly_static(
                "n_shape_params",
                [](const py::object& _) {
                    return SequenceConfig::n_shape_params();
                },
                "Size of shape (beta) space in dataset")
            .def_property_readonly_static(
                "n_body_joints",
                [](const py::object& _) {
                    return SequenceConfig::n_body_joints();
                },
                "Number of body joints in dataset")
            .def_property_readonly_static(
                "n_hand_joints",
                [](const py::object& _) {
                    return SequenceConfig::n_hand_joints();
                },
                "Number of hand joints in dataset")
            .def_property_readonly_static(
                "n_dmpls",
                [](const py::object& _) { return SequenceConfig::n_dmpls(); },
                "Number of DMPL parameters in dataset")
            .def_property_readonly(
                "empty", [](const SeqClass& obj) { return obj.n_frames == 0; })
            .def_property_readonly_static("has_dmpls",
                                          [](const py::object& _) {
                                              return SequenceConfig::n_dmpls() >
                                                     0;
                                          })

            .def_readonly("n_frames", &SeqClass::n_frames, "Number of frames")
            .def_readonly("frame_rate", &SeqClass::frame_rate)
            .def_readonly("gender", &SeqClass::gender,
                          "Gender (may be unknown)")
            .def_readonly("shape", &SeqClass::shape,
                          "Shape data (n_shape_params)")
            .def_readonly("trans", &SeqClass::trans,
                          "Translation data (n_frames, 3)")
            .def_readonly("pose", &SeqClass::pose,
                          "Pose data (n_frames, n_pose_params)")
            .def_readonly("dmpls", &SeqClass::dmpls,
                          "DMPLs data (n_frames, n_dmpls)")
            .def("__repr__", [](const SeqClass& obj) {
                return std::string("<smplxpp.Sequence(n_frames=") +
                       std::to_string(obj.n_frames) +
                       ", frame_rate=" + std::to_string(obj.frame_rate) + ")>";
            });
    declare_sequence_model_spec<SequenceConfig, model_config::SMPL>(cl);
    declare_sequence_model_spec<SequenceConfig, model_config::SMPLH>(cl);
    declare_sequence_model_spec<SequenceConfig, model_config::SMPLX>(cl);
    declare_sequence_model_spec<SequenceConfig, model_config::SMPLXpca>(cl);
}
}  // namespace

PYBIND11_MODULE(smplxpp, m) {
    m.doc() =
        R"pbdoc(SMPLXpp: SMPL/SMPL+H/SMPL-X implementation as C++ extension)pbdoc";
    m.attr("cuda") = CUDA_AVAILABLE;
    py::enum_<Gender>(m, "Gender")
        .value("unknown", Gender::unknown)
        .value("neutral", Gender::neutral)
        .value("female", Gender::female)
        .value("male", Gender::male);
    declare_model<model_config::SMPL>(m, "ModelS", "BodyS");
    declare_model<model_config::SMPLH>(m, "ModelH", "BodyH");
    declare_model<model_config::SMPLX>(m, "ModelX", "BodyX");
    declare_model<model_config::SMPLXpca>(m, "ModelXpca", "BodyXpca");

    declare_sequence<sequence_config::AMASS>(m, "SequenceAMASS");

    auto util = m.def_submodule("util");
    util.def("find_data_file", &util::find_data_file,
             "Path resolve helper: return a valid path to file in data/")
        .def("rodrigues", &util::rodrigues<float, Eigen::RowMajor>,
             "Rodrigues formula: convert axis-angle (3) to rotation "
             "matrix (3,3)")
        .def("mul_affine", &util::mul_affine<float, Eigen::RowMajor>,
             "Affine transform composition with bottom row omitted:"
             " a (3,4) x b (3,4) -> b in-place")
        .def("inv_affine", &util::inv_affine<float, Eigen::RowMajor>,
             "Affine transform in-place inversion")
        .def("inv_homogeneous", &util::inv_homogeneous<float, Eigen::RowMajor>,
             "Rigid-body transform in-place inversion")
        .def("gender_to_str", &util::gender_to_str, "Gender enum to string")
        .def("parse_gender", &util::parse_gender, "Gender enum from string");
}
