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
               float fovy,
               float aspect,
               float z_close,
               float z_far)
    : center_of_rot(center_of_rot), world_up(world_up),
      dist_to_center(dist_to_center),
      yaw(yaw), pitch(pitch),
      fovy(fovy), aspect(aspect), z_close(z_close), z_far(z_far)
{
    vectors_from_euler();
    update_proj();
    update_view();
}

void Camera::set_center(const Eigen::Ref<const Vector3f >& val) {
    center_of_rot = val;
    vectors_from_euler();
    update_view();
}
void Camera::set_world_up(const Eigen::Ref<const Vector3f >& val) {
    world_up = val.normalized();
    vectors_from_euler();
    update_view();
}
void Camera::set_dist(float val) {
    dist_to_center = val;
    vectors_from_euler();
    update_view();
}
void Camera::set_yaw(float val) {
    yaw = val;
    vectors_from_euler();
    update_view();
}
void Camera::set_pitch(float val) {
    pitch = val;
    vectors_from_euler();
    update_view();
}
void Camera::set_orientation(bool flip) {
    euler_up_orientation = flip ? -1.f : 1.f;
    vectors_from_euler();
    update_view();
}

void Camera::set_fovy(float val) {
    fovy = val;
    update_proj();
}
void Camera::set_aspect(float val) {
    aspect = val;
    update_proj();
}
void Camera::set_clip(float z_close_new, float z_far_new) {
    z_close = z_close_new;
    z_far = z_far_new;
    update_proj();
}

void Camera::rotate_with_mouse(float xoffset, float yoffset) {
    xoffset *= rotate_speed;
    yoffset *= rotate_speed;

    yaw   += xoffset * euler_up_orientation;
    pitch -= yoffset * euler_up_orientation;

    // Clamp pitch
    static const float PITCH_CLAMP = M_PI * 0.49999f;
    if (std::fabs(pitch) > PITCH_CLAMP) {
        pitch = PITCH_CLAMP * (pitch > 0.f ? 1.f : -1.f);
        yaw += M_PI;
        euler_up_orientation = -euler_up_orientation;
    }
    vectors_from_euler();
    update_view();
}

void Camera::pan_with_mouse(float xoffset, float yoffset) {
    xoffset *= pan_speed * dist_to_center;
    yoffset *= pan_speed * dist_to_center;

    center_of_rot += -xoffset * right + yoffset * up;
    vectors_from_euler();
    update_view();
}

void Camera::zoom_with_mouse(float amount) {
    if (amount < 0) dist_to_center *= 1.1;
    else dist_to_center *= 0.9;
    vectors_from_euler();
    update_view();
}

void Camera::reset_view() {
    center_of_rot.setZero();
    world_up = Vector3f(0.f, 1.f, 0.f);
    dist_to_center = 3.f;
    yaw = -M_PI/2;
    pitch = 0.0f;
    euler_up_orientation = 1.f;
    vectors_from_euler();
    update_view();
}

void Camera::vectors_from_euler() {
    front[0] = cos(yaw) * cos(pitch);
    front[1] = sin(pitch);
    front[2] = sin(yaw) * cos(pitch);
    pos = center_of_rot - front * dist_to_center;
    right = front.cross(euler_up_orientation * world_up).normalized();
    up = right.cross(front);
}

void Camera::update_view() {
    view = util::look_toward(pos, front, up);
}

void Camera::update_proj() {
    float tan_half_fovy = tan(fovy / 2.f);
    proj = util::persp(1.f / (tan_half_fovy * aspect), 1.f / tan_half_fovy, z_close, z_far);
    // calculate the new Front vector
    // Vector3f front;
    // front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    // front.y = sin(glm::radians(Pitch));
    // front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    // front = glm::normalize(front);
    // // also re-calculate the Right and Up vector
    // right = glm::normalize(glm::cross(Front, WorldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    // up    = glm::normalize(glm::cross(Right, Front));
}

}  // namespace meshview
