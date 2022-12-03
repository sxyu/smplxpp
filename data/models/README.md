## SMPL-X
1. Please go to http://smpl-x.is.tue.mpg.de and register
2. In downloads, click "download SMPL-X v...".
3. Open the downloaded zip and extract models/smplx/SMPLX_*.npz to ./smplx, so that we have: PROJ_ROOT/data/models/smplx/SMPLX_NEUTRAL.npz

You should now have
- `data/models/smplx/SMPLX_NEUTRAL.npz`
- `data/models/smplx/SMPLX_MALE.npz`
- `data/models/smplx/SMPLX_FEMALE.npz`

## SMPL
1. Please go to http://smpl.is.tue.mpg.de and register.
2. In Downloads, download the 'for Python users' zip.
3. Unzip to project root, so that there is a folder `smpl` on the same level as CMakeLists.txt
4. `cd` to the project root directory and run
 (for SMPL 1.0) `python tools/smpl2npz.py smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl` or
 (for SMPL 1.1) `python tools/smpl2npz.py smpl/models/basicModel_f_lbs_10_207_0_v1.1.0.pkl smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`

You should have
- `data/models/smpl/SMPL_MALE.npz`
- `data/models/smpl/SMPL_FEMALE.npz`

### Neutral
NOTE: as of SMPL 1.1, the neutral model is included in the SMPL release, and this is no longer needed

1. Please go to http://smplify.is.tue.mpg.de and register.
2. Download the zip under 'Code and model'.
3. Unzip to project root, so that there is a folder `smplify_public` on the same level as CMakeLists.txt
4. `cd` to the project root directory and run `python tools/smpl2npz.py smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`

You should have
- `data/models/smpl/SMPL_NEUTRAL.npz`

## SMPL+H
1. Please go to http://mano.is.tue.mpg.de and register
2. Download 'Extended SMPLH model for AMASS' in Downloads
3. `tar xf /path/to/smplh.tar.xz`
4. Manually copy gender/model.npz to data/models/smplh/SMPLH_GENDER.npz

You should have
- `data/models/smplh/SMPLH_NEUTRAL.npz`
- `data/models/smplh/SMPLH_MALE.npz`
- `data/models/smplh/SMPLH_FEMALE.npz`

## Notes
- SMPL/SMPL-H UV maps are provided by [github.com/radvani] in this issue
  https://github.com/Lotayou/densebody_pytorch/issues/4
- SMPL-X UV map is custom-made and likely not very good
- To use your own UV map, modify `data/models/<model>/uv.txt` UV map format:
    - First line: n = number of UV vertices
    - next n lines: u v = float coordinates of a vertex
    - next num_faces lines (num_faces is the number of rows in 'f' of the model):
      a b c = int, 1-based indices of UV vertices (from above) in each triangle
