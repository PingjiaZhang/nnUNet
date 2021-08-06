import os
from pathlib import Path
import glob
import pickle
import SimpleITK as sitk
import numpy as np

from nnunet.preprocessing.cropping import crop_to_bbox


def levelset_crop(nnUNet_base_path, task):
    raw_data_path = nnUNet_base_path / 'nnUNet_raw_data'
    cropped_data_path = nnUNet_base_path / 'nnUNet_cropped_data'

    pkl_files = sorted(glob.glob(str(cropped_data_path / task / '*.pkl')))
    for file_path in pkl_files:
        _, file = os.path.split(file_path)
        if file == 'dataset_properties.pkl':
            continue
        subj = file[:-4]

        pkl_path = cropped_data_path / task / file
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)

        # npz_path = cropped_data_path / task / (subj + '.npz')
        # npz_data = np.load(npz_path)['data']

        # img_path = raw_data_path / task / 'imagesTr' / (subj + '_0000.nii.gz')
        # img = sitk.ReadImage(str(img_path))
        # img_np = sitk.GetArrayFromImage(img)

        bbox = pkl_data['crop_bbox']
        # img_crop_np = crop_to_bbox(img_np, bbox)
        # print(np.all(img_crop_np == npz_data[0]))

        levelset_path = raw_data_path / task / 'labelsTr_levelset' / (subj + '.nii.gz')
        levelset = sitk.ReadImage(str(levelset_path))
        levelset_np = sitk.GetArrayFromImage(levelset)
        levelset_crop_np = -crop_to_bbox(levelset_np, bbox) / 3

        levelet_npz_path = cropped_data_path / task / 'gt_levelset' / (subj + '_levelset.npz')
        np.savez_compressed(levelet_npz_path, levelset=levelset_crop_np)


if __name__ == '__main__':
    nnUNet_base_path = Path('/home/weiwei/workdata/DeepPrep/nnUNet')

    task = 'Task600_SurfRecon'

    levelset_crop(nnUNet_base_path, task)
