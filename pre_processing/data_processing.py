import os
import glob
import json
import numpy as np
from random import randint
from util import transfer
from util import variable


def data_process(data_type=0, skip_list=None, special_func=None):
    if not os.path.isdir(variable.RAW_DIR):
        os.mkdir(variable.RAW_DIR)
        return
    if not os.path.isdir(variable.DATA_DIR):
        os.mkdir(variable.DATA_DIR)
    if not os.path.isdir(variable.TRAIN_DIR):
        os.mkdir(variable.TRAIN_DIR)
    if not os.path.isdir(variable.TEST_DIR):
        os.mkdir(variable.TEST_DIR)
    if not os.path.isdir(variable.PREDICT_DIR):
        os.mkdir(variable.PREDICT_DIR)
    dir_list = glob.glob(variable.RAW_DIR + '*')
    for dir0 in dir_list:
        if os.path.isdir(dir0):
            dir_name = os.path.basename(dir0)
            file_list = glob.glob(variable.RAW_DIR + dir_name + '/*.dat')
            for file in file_list:
                file_name = os.path.basename(file)
                with open(variable.RAW_DIR + dir_name + '/' + file_name, 'r') as file_opened:
                    file_json = json.load(file_opened)
                name_list = ['T1', 'T2', 'FIFO1', 'FIFO2', 'FIFO3']
                image_list = transfer.structure_transfer(file_json, name_list)
                x = np.asarray(image_list, dtype=np.float32)
                if data_type < 3:
                    y = np.asarray(file_json['RESULT'], dtype=np.uint8)

                # skip result not need
                if skip_list and np.where(y == 1)[0][0] in skip_list:
                    continue
                if data_type < 3:
                    data = {
                        'x': x.tolist(),
                        'y': result_filter(y.tolist(), skip_list, special_func),
                    }
                    result = data['y'].index(1)
                    if not os.path.isdir(variable.TEST_DIR + str(result)):
                        os.mkdir(variable.TEST_DIR + str(result))
                    if not os.path.isdir(variable.TRAIN_DIR + str(result)):
                        os.mkdir(variable.TRAIN_DIR + str(result))
                else:
                    data = {
                        'x': x.tolist()
                    }
                # save data
                if data_type == 0:
                    if randint(0, 9) == 0:
                        with open(variable.TEST_DIR + str(result) + '/' + dir_name + '.' + file_name, 'w') as test_file:
                            json.dump(data, test_file)
                    else:
                        with open(variable.TRAIN_DIR + str(result) + '/' + dir_name + '.' + file_name, 'w') as train_file:
                            json.dump(data, train_file)
                elif data_type == 1:
                    with open(variable.TRAIN_DIR + str(result) + '/' + dir_name + '.' + file_name, 'w') as train_file:
                        json.dump(data, train_file)
                elif data_type == 2:
                    with open(variable.TEST_DIR + str(result) + '/' + dir_name + '.' + file_name, 'w') as test_file:
                        json.dump(data, test_file)
                else:
                    with open(variable.PREDICT_DIR + '/' + dir_name + '.' + file_name, 'w') \
                            as predict_file:
                        json.dump(data, predict_file)


def result_filter(result_list, skip_list, special_func=None):
    if skip_list:
        for skip in sorted(skip_list, reverse=True):
            del result_list[skip]
    if special_func:
        result_list = special_func(result_list)
    return result_list
