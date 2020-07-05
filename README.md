SMPL-X and CAPE C++ implementation in Eigen and CUDA.

## Dependencies
- Compiler supporting C++ 14
- OpenGL 3+ (For viewer)

### Optional Dependencies
- CUDA (I have 11)
- OpenMP for CPU parallelization (currently not actually used)

### Vendored 3rd party libraries
The following dependencies are included in the repo and don't need to be installed
- cnpy (for npy/npz I/O) https://github.com/rogersce/cnpy
    - zlib: needed by cnpy to read npz (uses system zlib if available)
- Eigen 3 (I have 3.3.7) http://eigen.tuxfamily.org/
    (uses system zlib if available, unless CMake option USE_SYSTEM_EIGEN=OFF)
- The following are used only if viewer is build
    - glew 2
    - glfw 3.3
    - ImGui
    - stb_image

## Building
- To configure, `mkdir build && cd build && cmake ..`
    - To disable the OpenGL Viewer, replace the above cmake command with `cmake .. -D BUILD_VIEWER=OFF`
- To build, use `make -j<number-of threads-here>` on unix-like systems,
    `cmake --build . --config Release` else
- To install (unix only), use `sudo make install`

### Outputs
- `smpl_example`: Writes SMPL-X model to`out.obj`
- `smpl_viewer` (only if `BUILD_VIEWER=ON` in CMake): Shows an interactive 3D viewer, including parameter controls
