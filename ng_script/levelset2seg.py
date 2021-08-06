import os
import ants
from pathlib import Path
import numpy as np


def levelset2seg(raw_data_path, task):
    levelset_files = sorted(os.listdir(raw_data_path / task / 'labelsTr_levelset'))
    os.makedirs(raw_data_path / task / 'labelsTr', exist_ok=True)
    for file in levelset_files:
        _, file_name = os.path.split(file)
        file_path = raw_data_path / task / 'labelsTr_levelset' / file
        levelset = ants.image_read(str(file_path))
        levelset_np = levelset.numpy()
        idx = (levelset_np < 3) & (levelset_np > -3)
        mask_np = np.zeros(levelset_np.shape, dtype=np.float32)
        mask_np[idx] = 1
        mask = ants.from_numpy(mask_np, levelset.origin, levelset.spacing, levelset.direction)
        save_file = raw_data_path / task / 'labelsTr' / file_name
        ants.image_write(mask, str(save_file))
        print(file)


if __name__ == '__main__':
    nnUNet_base_path = Path('/home/weiwei/workdata/DeepPrep/nnUNet')

    raw_data_path = nnUNet_base_path / 'nnUNet_raw_data'

    task = 'Task600_SurfRecon'

    levelset2seg(raw_data_path, task)
