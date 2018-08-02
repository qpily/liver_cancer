from util import file_processing
from util import transfer
import os
import json
import numpy as np


def execute(folder_name, folder_list=None):
    file_list = file_processing.read_file(folder_name, folder_list)
    for file in file_list:
        file_name = os.path.splitext(os.path.basename(file))[0]
        with open(file, 'r') as file_opened:
            file_json = json.load(file_opened)
        x = np.asarray(file_json['x'], dtype=np.float32)
        y = np.asarray(file_json['y'], dtype=np.uint8)

        # rotation
        x_rot1 = transfer.rotate(x, 1)
        x_rot2 = transfer.rotate(x, 2)
        x_rot3 = transfer.rotate(x, 3)
        dir_path = os.path.dirname(file)
        save_augmentation(dir_path, file_name, 'rot1', x_rot1, y)
        save_augmentation(dir_path, file_name, 'rot2', x_rot2, y)
        save_augmentation(dir_path, file_name, 'rot3', x_rot3, y)

        # flip
        x_flip0 = transfer.flip(x, 0)
        x_flip1 = transfer.flip(x, 1)
        save_augmentation(dir_path, file_name, 'flip0', x_flip0, y)
        save_augmentation(dir_path, file_name, 'flip1', x_flip1, y)

        # mix
        x_mix0 = transfer.flip(x_rot1, 0)
        x_mix1 = transfer.flip(x_rot1, 1)
        save_augmentation(dir_path, file_name, 'mix0', x_mix0, y)
        save_augmentation(dir_path, file_name, 'mix1', x_mix1, y)


def save_augmentation(dir_path, file_name, suffix, x, y):
    data = {
        'x': x.tolist(),
        'y': y.tolist()
    }
    with open(dir_path + '/' + file_name + '.' + suffix + '.dat', 'w') as file:
        json.dump(data, file)
