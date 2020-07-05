#include "meshview/camera.hpp"

#include <iostream>
#include <Eigen/Geometry>

#include "meshview/util.hpp"

namespace meshview {

Camera::Camera(const Vector3f& center_of_rot,
               const Vector3f& world_up,
               float dist_to_center,
               float yaw,
               float pitch,
               float roll,
               float fovy,
               float aspect,
               float z_close,
               float z_far)
    : center_of_rot(center_of_rot), world_up(world_up),
      dist_to_center(dist_to_center),
      yaw(yaw), pitch(pitch), roll(roll),
      fovy(fovy), aspect(aspect), z_close(z_close), z_far(z_far)
{
    update_proj();
    update_view();
}

void Camera::rotate_with_mouse(float xoffset, float yoffset) {
    xoffset *= rotate_speed;
    yoffset *= rotate_speed;

    float cr = cos(roll), sr = sin(roll);
    yaw   += xoffset * cr + yoffset * sr;
    pitch -= yoffset * cr + xoffset * sr;

    // Clamp pitch
    static const float PITCH_CLAMP = M_PI * 0.49999f;
    if (std::fabs(pitch) > PITCH_CLAMP) {
        pitch = PITCH_CLAMP * (pitch > 0.f ? 1.f : -1.f);
        yaw = yaw += M_PI;
        roll = roll += M_PI;
    }
    update_view();
}

void Camera::roll_with_mouse(float xoffset, float yoffset) {
    xoffset *= rotate_speed;
    roll += xoffset;

    update_view();
}

void Camera::pan_with_mouse(float xoffset, float yoffset) {
    xoffset *= pan_speed * dist_to_center;
    yoffset *= pan_speed * dist_to_center;

    center_of_rot += -xoffset * right + yoffset * up;
    update_view();
}

void Camera::zoom_with_mouse(float amount) {
    if (amount < 0) dist_to_center *= scroll_factor;
    else dist_to_center *= 1.f / scroll_factor;
    update_view();
}

void Camera::reset_view() {
    center_of_rot.setZero();
    world_up = Vector3f(0.f, 1.f, 0.f);
    dist_to_center = 3.f;
    yaw = -M_PI/2;
    pitch = roll = 0.0f;
    update_view();
}

void Camera::update_view() {
    front[0] = cos(yaw) * cos(pitch);
    front[1] = sin(pitch);
    front[2] = sin(yaw) * cos(pitch);
    pos = center_of_rot - front * dist_to_center;
    Eigen::AngleAxisf aa_roll(roll, front);
    right = front.cross(aa_roll * world_up).normalized();
    up = right.cross(front);
    view = util::look_toward(pos, front, up);
}

void Camera::update_proj() {
    float tan_half_fovy = tan(fovy / 2.f);
    proj = util::persp(1.f / (tan_half_fovy * aspect), 1.f / tan_half_fovy, z_close, z_far);
}

}  // namespace meshview
