#pragma once
#ifndef VIEWER_CAMERA_B9A22093_7B3A_49A5_B013_8935B9AEAEE6
#define VIEWER_CAMERA_B9A22093_7B3A_49A5_B013_8935B9AEAEE6

#include <Eigen/Core>

#include <cmath>

#include "meshview/common.hpp"

namespace meshview {

class Camera {
    public:
        // Construct camera with given params
        explicit Camera(
               const Vector3f& center_of_rot = Eigen::Vector3f(0.f, 0.f, 0.f),
               const Vector3f& world_up = Eigen::Vector3f(0.f, 1.f, 0.f),
               float dist_to_center = 3.f,
               float yaw = -M_PI/2,
               float pitch = 0.0f,
               float fovy = M_PI / 4.f,
               float aspect = 5.f / 3.f,
               float z_close = 0.1f,
               float z_far = 1e3f);

        // Get camera position
        inline Vector3f get_pos() const { return pos; }

        // * View *
        // Set center of rotation
        void set_center(const Eigen::Ref<const Vector3f >& val);
        // Set world up direction; changes how mouse rotation x/y behaves
        void set_world_up(const Eigen::Ref<const Vector3f >& val);
        // Set distance to center of rotation (zoom)
        void set_dist(float dist);
        // Set yaw
        void set_yaw(float yaw);
        // Set pitch
        void set_pitch(float yaw);
        // Set 'orientation' (vertical inversion of world_up,
        // to allow 360-degree rotation with euler angles)
        void set_orientation(bool flip);

        // * Projection *
        // Set vertical field-of-view in radians (default PI/4 = 45 degs)
        void set_fovy(float val);
        // Set width/height ratio of viewport (default 5/3)
        void set_aspect(float val);
        // Set clipping distances (default 0.1, 1000)
        void set_clip(float z_close_new, float z_far_new);

        // Handlers
        void rotate_with_mouse(float xoffset, float yoffset);
        void pan_with_mouse(float xoffset, float yoffset);
        void zoom_with_mouse(float amount);

        // Reset the view
        void reset_view();

        // Camera matrices
        Matrix4f view;
        Matrix4f proj;

        // Camera control options
        float pan_speed = .0015f;
        float rotate_speed = .008f;

    private:
        // View parameters
        Vector3f pos, center_of_rot;
        Vector3f front, up, right, world_up; // right only used for euler angles
        float dist_to_center;

        // Euler angles
        float yaw, pitch;

        // Projection parameters
        float fovy, aspect, z_close, z_far;

        // Either 1 or -1: if -1 then the camera is upside-down
        // (only used for euler angles)
        float euler_up_orientation = 1.f;

        // Compute vectors from euler angles + world_up
        void vectors_from_euler();

        // Update view matrix
        void update_view();

        // Update proj matrix
        void update_proj();
};

}  // namespace meshview

#endif  // ifndef VIEWER_CAMERA_B9A22093_7B3A_49A5_B013_8935B9AEAEE6
