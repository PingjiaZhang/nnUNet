import os
from pathlib import Path
import glob
import pickle
import SimpleITK as sitk
import numpy as np
from nnunet.training.network_training import nnUNetTrainerV2
from nnunet.run.default_configuration import get_default_configuration

if __name__ == '__main__':
    nnUNet_base_path = Path('/home/weiwei/workdata/DeepPrep/nnUNet')

    task = 'Task600_SurfRecon'

    network = '3d_fullres'
    network_trainer = 'nnUNetTrainerV2'
    plans_identifier = 'nnUNetPlansv2.1'

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)
    levelset_crop(nnUNet_base_path, task)
