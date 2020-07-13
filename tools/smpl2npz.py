from __future__ import print_function
import numpy as np
import pickle
import sys, os
from os import path

if len(sys.argv) < 2:
    print(
        'Util to convert SMPL pkl model to npz usable by our C++ software. Outputs to data/models/smpl'
    )
    print(
        'Usage: python tools/smpl2npz.py /path/to/smpl/models/basic[mM]odel_[mf]_lbs_10_207_0_v1.0.0.pkl [/path/to/other.pkl...]'
    )
    print('IMPORTANT: Run from project root dir (the one containing tools)')
    print(
        'Tip: Download SMPL m/f models from https://smpl.is.tue.mpg.de/ ("for python users")'
    )
    print(
        'Tip: Download SMPL neutral model from https://smplify.is.tue.mpg.de/ ("for python users")'
    )
output_dir = 'data/models/smpl/'
os.makedirs(output_dir, exist_ok=True)

for model_path in sys.argv[1:]:
    with open(model_path, 'rb') as model_file:
        try:
            model_data = pickle.load(model_file, encoding='latin1')
        except:
            model_data = pickle.load(model_file)

        output_data = {}
        for key, data in model_data.items():
            dtype = str(type(data))
            if 'chumpy' in dtype:
                # Convert chumpy
                output_data[key] = np.array(data)
            elif 'scipy.sparse' in dtype:
                # Convert scipy sparse matrix
                output_data[key] = data.toarray()
            else:
                output_data[key] = data
        model_fname = path.split(model_path)[1]
        if len(model_fname) > 11 and model_fname[11] == 'f':
            output_gen = 'FEMALE'
        elif len(model_fname) > 11 and model_fname[11] == 'm':
            output_gen = 'MALE'
        else:
            output_gen = 'NEUTRAL'
        output_path = path.join(output_dir, 'SMPL_' + output_gen + '.npz')
        print('Writing', output_path)
        np.savez_compressed(output_path, **output_data)
