SMPL-X and CAPE C++ implementation in Eigen and CUDA.

## Dependencies
- Compiler supporting C++ 14
- OpenGL 3+

### Optional Dependencies
- CUDA (I have 11)
- OpenMP for CPU parallelization (currently not actually used)

### Vendored 3rd party libraries
The following dependencies are included in the repo and don't need to be installed
- cnpy (for npy/npz I/O) https://github.com/rogersce/cnpy
    - zlib: needed by cnpy to read npz (uses system zlib if available)
- glew 2
- glfw 3.3
- Eigen 3 (I have 3.3.7) http://eigen.tuxfamily.org/
    (uses system zlib if available, unless CMake option USE_SYSTEM_EIGEN=OFF)
