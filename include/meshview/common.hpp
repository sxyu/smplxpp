#pragma once
#ifndef VIEWER_COMMON_93D99C8D_E8CA_4FFE_9716_D8237925F910
#define VIEWER_COMMON_93D99C8D_E8CA_4FFE_9716_D8237925F910

#include <Eigen/Core>

namespace meshview {

using Matrix2f = Eigen::Matrix2f;
using Matrix3f = Eigen::Matrix3f;
using Matrix4f = Eigen::Matrix4f;
using Vector2f = Eigen::Vector2f;
using Vector3f = Eigen::Vector3f;
using Vector4f = Eigen::Vector4f;


using Scalar = float;
using MeshIndex = uint32_t;
using PointCloud = Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor>;
using PointCloudUVN = Eigen::Matrix<Scalar, Eigen::Dynamic, 8, Eigen::RowMajor>;
using PointCloud2D = Eigen::Matrix<Scalar, Eigen::Dynamic, 2, Eigen::RowMajor>;

using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using MatrixColMajor = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

using Triangles = Eigen::Matrix<MeshIndex, Eigen::Dynamic, 3, Eigen::RowMajor>;

}

#endif  // ifndef VIEWER_COMMON_93D99C8D_E8CA_4FFE_9716_D8237925F910
