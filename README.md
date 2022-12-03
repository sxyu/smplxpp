SMPL-X C++ implementation in Eigen and CUDA. (WIP!)
AMASS integration included.

## Dependencies
- Compiler supporting C++ 17 (I am using GCC 9.3.0)

### Optional Dependencies
- OpenGL 3+ (Required to build the viewer)
- CUDA Toolkit with nvcc supporting C++ 14 (I have CUDA 11, not tested with earlier versions)
- Very optional: https://www.github.com/sxyu/sdf to build 'smplx-prox', which shows a cross-section of continuous SDF values inside the SMPL model

## Getting the SMPL/SMPL+H/SMPL-X Models
- See [data/models/README.md](https://github.com/sxyu/smplxpp/tree/master/data/models)
- Set `SMPLX_DIR` environment variable to the project root directory to allow finding
  default data files (else you must `cd` to project root or in a subdirectory for the
  project to work)

## Building
Note: I have built successfully on Ubuntu 20.04 and Windows 10 (VS2019). The project appears to run about 1.5x slower on Windows for some reason.

- To configure, `mkdir build && cd build && cmake ..`
    - To disable the OpenGL Viewer, replace the above cmake command with `cmake .. -D SMPLX_BUILD_VIEWER=OFF`
- To build, use `make -j<number-of threads-here>` on unix-like systems,
    `cmake --build . --config Release` else
- To install (unix only), use `sudo make install` (TODO: add CMake find module)

### Python
To install Python bindings, use `pip install .`
You may need to
 first install pybind11 from https://github.com/pybind/pybind11.
Usage example:
```python
from smplxpp import ModelS, BodyS, SequenceAMASS, Gender
# SMPL model. Use ModelX/BodyX for SMPL-X,
# ModelH/BodyH for SMPL+H, ModelXpca/BodyXPca for SMPL-X with hand PCA
model = ModelS(Gender.male)
body = BodyS(model)

seq = SequenceAMASS("/data/AMASS/KIT/10/RightTurn01_poses.npz")

# Use sequence to pose body
seq.set_shape(body)
seq.set_pose(body, 10) # Frame 10

# Do skinning (prefers GPU)
body.update()

print(body.verts) # vertices

import trimesh
tm = trimesh.Trimesh(body.vertices, model.faces) 
tm.show()
```

## Example programs
- `smplx-example`: Writes SMPL-X model to`out.obj`
    - Usage: `./smplx-example gender` where gender (optional, case insensitive)
      should be NEUTRAL/MALE/FEMALE; NEUTRAL is default
- `smplx-viewer` (if `SMPLX_BUILD_VIEWER=ON` in CMake):
   Shows an interactive 3D viewer, including parameter controls
    - Usage: `./smplx-viewer model gender device poseblends where
        - All arguments are position and optional
        - model may be S/S1/H/X/Z/Xp/Zp (sorry about the mess here), where S means SMPL, H means SMPL+H, X means SMPL-X (with hand joints), and Xp means SMPL-X (with hand PCA). Default is S
            - New: X/Xp now specify SMPL-X v1.1 by default. Use Z/Zp for v1.0
            - New: S now specify SMPL v1.1 by default. Use S1 for v1.0
        - gender may be NEUTRAL/MALE/FEMALE; NEUTRAL is default (case insensitive)
        - device may be gpu/cpu; gpu is default and will fallback to cpu automatically
        - poseblends may be on/off, default on; off turns off pose blendshapes, which
          speeds up computation dramatically
        - Example: `./smplx-viewer X MALE`, `./smplx-viewer H FEMALE`
        - `./smplx-viewer` will open plain neutral SMPL model (if available)
- `smplx-amass`: AMASS viewer
    - Usage: `./smplx-amass model npz_path`
        - All arguments are position and optional
        - model may be `S/H/X`, where S means SMPL, H means SMPL+H, X means SMPL-X. Note Xp (hand PCA) is not available for AMASS integration. Default **H**
        - npz_path: optionally, path to AMASS .npz to load on open
        - `./smplx-amass` opens a blank viewer with option to browse for and load a npz
- `smplx-sdf`: View SMPL SDF

## Library usage
- TBA, refer to examples (`main_*.cpp`) for now

## License
This library is licensed under Apache v2 (non-copyleft).
However, remember that the following are non-commercial research-only:
- SMPL
- SMPL-X
- Mano/SMPL+H
- AMASS dataset
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

<a id="5">[5]</a> AMASS: https://amass.is.tue.mpg.de/.
AMASS: Archive of Motion Capture as Surface Shapes.
Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.


### Vendored 3rd party libraries
The following dependencies are included in the repo and don't need to be installed
- cnpy (for npy/npz I/O) https://github.com/rogersce/cnpy
    - zlib: needed by cnpy to read npz (uses system zlib if available)
- Eigen *3.3.90* http://eigen.tuxfamily.org/
    - Note this is NEWER than the latest release! I had problems getting 3.3.7 to work with CUDA, so I vendored the version on master. You are welcome to try again.
- The following are used only if building viewer:
    - glew 2
    - glfw 3.3
    - Dear ImGui, with modifications
        - ImGui filebrowser https://github.com/AirGuanZ/imgui-filebrowser, with modifications
    - stb_image.h
