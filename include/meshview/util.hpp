#pragma once
#ifndef VIEWER_UTIL_67A492E2_6CCA_4FA8_9763_90A5DA4F6837
#define VIEWER_UTIL_67A492E2_6CCA_4FA8_9763_90A5DA4F6837

#include <string>
#include "common.hpp"

namespace meshview {
namespace util {

// Perspective projection matrix
// xscale/yscale: scaling on x/y axes
// z_near/z_far: clip distances
Matrix4f persp(float xscale, float yscale, float z_near, float z_far);

// Make look at matrix
// pos: camera position
// fw: forward direction (normalized)
// up: up direction for camera (normalized)
Matrix4f look_toward(const Eigen::Ref<const Vector3f>& pos,
                     const Eigen::Ref<const Vector3f>& fw,
                     const Eigen::Ref<const Vector3f>& up);


// Estimate normals given points in pointcloud with given *triangles*
// outputs normals into out. If faces is empty, same as 'no element buffer' version.
// NOTE: out must already be of same size as verts
void estimate_normals(const Eigen::Ref<const PointCloud>& verts,
                      const Eigen::Ref<const Triangles>& faces,
                      Eigen::Ref<PointCloud> out);

// Estimate normals given points in pointcloud with no element buffer
// (point 0,1,2 are 1st triangle, 3,4,5 2nd, etc..)
// outputs normals into out.
// NOTE: out must already be of same size as verts
void estimate_normals(const Eigen::Ref<const PointCloud>& verts,
                      Eigen::Ref<PointCloud> out);

// Path resolve helper
std::string find_data_file(const std::string& data_path);

}  // namespace util
}  // namespace meshview

#endif  // ifndef VIEWER_UTIL_67A492E2_6CCA_4FA8_9763_90A5DA4F6837
