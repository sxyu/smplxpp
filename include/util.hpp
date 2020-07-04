#pragma once
#ifndef SMPL_UTIL_63B0803D_E0C7_4529_A796_9F6ED269E89F
#define SMPL_UTIL_63B0803D_E0C7_4529_A796_9F6ED269E89F

#define _SMPL_ASSERT(x) do { \
    auto tx = (x); \
    if (!tx) { \
    std::cout << "Assertion FAILED: \"" << #x << "\" (" << tx << \
        ")\n  at " << __FILE__ << " line " << __LINE__ <<"\n"; \
    std::exit(1); \
}} while(0)
#define _SMPL_ASSERT_EQ(x, y) do {\
    auto tx = x; auto ty = y; \
    if (tx != ty) { \
    std::cout << "Assertion FAILED: " << #x << " != " << #y << " (" << tx << " != " << ty << \
        ")\n  at " << __FILE__ << " line " << __LINE__ <<"\n"; \
    std::exit(1); \
}} while(0)

#include <chrono>
#define _SMPL_BEGIN_PROFILE auto start = std::chrono::high_resolution_clock::now()
#define _SMPL_PROFILE(x) do{printf("%s: %f ns\n", #x, std::chrono::duration<double, std::nano>(std::chrono::high_resolution_clock::now() - start).count()); start = std::chrono::high_resolution_clock::now(); }while(false)
#define _SMPL_PROFILE_STEPS(x,stp) do{printf("%s: %f ns / step\n", #x, std::chrono::duration<double, std::nano>(std::chrono::high_resolution_clock::now() - start).count()/(stp)); start = std::chrono::high_resolution_clock::now(); }while(false)

#include<Eigen/Core>
#include<Eigen/Geometry>

namespace smpl {
namespace util {

// Angle-axis to rotation matrix using custom implementation
template <class T, int Option = Eigen::ColMajor>
inline Eigen::Matrix<T, 3, 3, Option> rodrigues(const Eigen::Ref<Eigen::Vector<T, 3> >& vec) {
    Eigen::Matrix3d d;
    const T theta = vec.norm();
    const Eigen::Matrix<T, 3, 3, Option> eye = Eigen::Matrix<T, 3, 3, Option>::Identity();

    if (std::fabs(theta) < 1e-5f) return eye;
    else {
        const T c = std::cos(theta);
        const T s = std::sin(theta);
        const T itheta = T(1) / theta;
        const Eigen::Matrix<T, 3, 1> r = vec / theta;
        Eigen::Matrix<T, 3, 3, Option> skew;
        skew << 0, -r.z(), r.y(),
                r.z(), 0, -r.x(),
                -r.y(), r.x(), 0;
        return c * eye + (1 - c) * r * r.transpose() + s * skew;
    }
}

// Angle-axis to rotation matrix through Eigen quaternion
// (slightly slower than rodrigues, not useful)
template <class T, int Option = Eigen::ColMajor>
inline Eigen::Matrix<T, 3, 3, Option> rodrigues_eigen(const Eigen::Ref<Eigen::Vector<T, 3> >& vec) {
    return Eigen::template AngleAxis<T>(vec.norm(), vec / vec.norm()).toRotationMatrix();
}

// Affine transformation matrix (hopefully) faster multiplication
// bottom row omitted
template <class T, int Option = Eigen::ColMajor>
inline void mul_affine(const Eigen::Ref<const Eigen::Matrix<T, 3, 4, Option> >& a,
            Eigen::Ref<Eigen::Matrix<T, 3, 4, Option> > b) {
    b.template leftCols<3>() = a.template leftCols<3>() * b.template leftCols<3>();
    b.template rightCols<1>() = a.template rightCols<1>() +
        a.template leftCols<3>() * b.template rightCols<1>();
}

// Path resolve helper
std::string find_data_file(const std::string& data_path);

}  // namespace util
}  // namespace smpl

#endif  // ifndef SMPL_UTIL_63B0803D_E0C7_4529_A796_9F6ED269E89F
