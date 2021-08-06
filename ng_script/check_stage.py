import os
from pathlib import Path
import glob
import numpy as np

if __name__ == '__main__':
    nnUNet_base_path = Path('/home/weiwei/workdata/DeepPrep/nnUNet')

    task = 'Task600_SurfRecon'

    cropped_data_path = nnUNet_base_path / 'nnUNet_cropped_data'
    nnUNet_preprocessed_path = nnUNet_base_path / 'nnUNet_preprocessed'

    npz_files = sorted(glob.glob(str(cropped_data_path / task / '*.npz')))

    for file_path in npz_files:
        _, file = os.path.split(file_path)
        subj = file[:-4]
        npz_path = cropped_data_path / task / (subj + '.npz')
        npz_data = np.load(npz_path)['data']

        npz_pp_path = nnUNet_preprocessed_path / task / 'nnUNetData_plans_v2.1_stage0' / (subj + '.npz')
        npz_pp_data = np.load(npz_pp_path)['data']

        print(np.all(npz_data[1] == npz_pp_data[1]))
