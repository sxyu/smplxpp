SMPL-X C++ implementation in Eigen and CUDA. (WIP!)

## Dependencies
- Compiler supporting C++ 14 (I am using GCC 9.3.0)

### Optional Dependencies
- OpenGL 3+ (Required to build the viewer)
- CUDA Toolkit, supporing C++ 14 (I have CUDA 11, not tested with earlier versions)

## Getting the SMPL/SMPL+H/SMPL-X Models
- See [data/models/README.md](https://github.com/sxyu/smplxpp/tree/master/data/models)

## Building
Note: I have only tried building on Ubuntu 20.04. However, it should not be hard to build the project in other systems, since it is fairly lightweight.

- To configure, `mkdir build && cd build && cmake ..`
    - To disable the OpenGL Viewer, replace the above cmake command with `cmake .. -D SMPLX_BUILD_VIEWER=OFF`
- To build, use `make -j<number-of threads-here>` on unix-like systems,
    `cmake --build . --config Release` else
- To install (unix only), use `sudo make install` (TODO: add CMake find module)

## Usage
- `smplx-example`: Writes SMPL-X model to`out.obj`
    - Usage: `./smplx-example [GENDER]` where GENDER should be NEUTRAL/MALE/FEMALE; NEUTRAL is default (case insensitive)
- `smplx-viewer` (if `SMPLX_BUILD_VIEWER=ON` in CMake):
   Shows an interactive 3D viewer, including parameter controls
    - Usage: `smplx-viewer [MODEL [GENDER]]` where
        - MODEL should be S/H/X, where S means SMPL, H means SMPL+H, X means SMPL-X
        - GENDER should be NEUTRAL/MALE/FEMALE; NEUTRAL is default (case insensitive)
        - Example: `./smplx-viewer X MALE`, `./smplx-viewer H FEMALE`

## Library usage
- TBA, refer to examples (`main_*.cpp`) for now

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
