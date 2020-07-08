SMPL-X and CAPE C++ implementation in Eigen and CUDA. (WIP!)

## Dependencies
- Compiler supporting C++ 14
- OpenGL 3+ (For viewer)

### Optional Dependencies
- CUDA (I have 11) supporing C++ 14
- OpenMP for CPU parallelization (currently not actually used)

## Models
- See how to get and install SMPL/SMPL-X/SMPL+H in `data/models`

## Building
- To configure, `mkdir build && cd build && cmake ..`
    - To disable the OpenGL Viewer, replace the above cmake command with `cmake .. -D SMPLX_BUILD_VIEWER=OFF`
- To build, use `make -j<number-of threads-here>` on unix-like systems,
    `cmake --build . --config Release` else
- To install (unix only), use `sudo make install`

### Outputs
- `smpl_example`: Writes SMPL-X model to`out.obj`
- `smpl_viewer` (only if `SMPLX_BUILD_VIEWER=ON` in CMake): Shows an interactive 3D viewer, including parameter controls

## License
This library is licensed under Apache v2 (non-copyleft).
However, remember that the following are non-commercial research-only:
- SMPL
- SMPL-X
- SMPL+H
- SMPLify

## References
<a id="1">[1]</a> SMPL: http://smpl.is.tue.mpg.de.
SMPL: A Skinned Multi-Person Linear Model.  Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, Michael J. Black. 2015
<a id="2">[2]</a> SMPLify: http://smplify.is.tue.mpg.de.
Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image.
Federica Bogo*, Angjoo Kanazawa*, Christoph Lassner, Peter Gehler, Javier Romero, Michael Black.
2016
<a id="3">[3]</a> Mano: http://mano.is.tue.mpg.de.
Embodied Hands: Modeling and Capturing Hands and Bodies Together. Javier Romero*, Dimitrios Tzionas*, and Michael J Black. 2017
<a id="4">[4]</a> SMPL-X: http://smpl-x.is.tue.mpg.de.
Expressive Body Capture: 3D Hands, Face, and Body from a Single Image. G. Pavlakos*, V. Choutas*, N. Ghorbani, T. Bolkart, A. A. A. Osman, D. Tzionas and M. J. Black. 2019


### Vendored 3rd party libraries
The following dependencies are included in the repo and don't need to be installed
- cnpy (for npy/npz I/O) https://github.com/rogersce/cnpy
    - zlib: needed by cnpy to read npz (uses system zlib if available)
- Eigen 3 (I have 3.3.7) http://eigen.tuxfamily.org/
    (uses system zlib if available, unless CMake option USE_SYSTEM_EIGEN=OFF)
- The following are used only if building viewer:
    glew 2, glfw 3.3, ImGui, stb_image
