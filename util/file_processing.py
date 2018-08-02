import glob
import os
import json
import numpy as np
from keras.models import model_from_json
from util import variable, pyplot


def read_data(mode='train', folder_list=None):
    file_list = read_file(mode, folder_list)
    data = {
        'x': [],
        'y': [],
        'name_list': []
    }
    for file in file_list:
        filename = os.path.basename(file)
        with open(file, 'r') as file_opened:
            file_json = json.load(file_opened)
        x = np.asarray(file_json['x'], dtype=np.float32)
        data['x'].append(x)
        if not mode == 'predict':
            y = np.asarray(file_json['y'], dtype=np.uint8)
            data['y'].append(y)
            # add count
            if not data.get('count'):
                data['count'] = [0] * y.size
            data['count'][np.where(y == 1)[0][0]] += 1
        data['name_list'].append(filename)

    data['x'] = np.asarray(data['x'])
    if not mode == 'predict':
        data['y'] = np.asarray(data['y'])
    else:
        del data['y']
    return data


def read_file(folder_name, folder_list=None):
    if not os.path.isdir(variable.DATA_DIR + folder_name):
        raise Exception('No ' + folder_name + ' data')
    dir_list = glob.glob(variable.DATA_DIR + folder_name + '/*')
    file_list = []
    if len(dir_list) > 0:
        for dir_file in dir_list:
            if os.path.isdir(dir_file):
                dir_name = os.path.basename(dir_file)
                if (folder_list and dir_name in folder_list) or not folder_list:
                    file_in_dir = glob.glob(variable.DATA_DIR + folder_name + '/' + dir_name + '/*.dat')
                    file_list.extend(file_in_dir)
            elif os.path.splitext(os.path.basename(dir_file))[1] == '.dat':
                file_list.append(dir_file)
    return file_list


def read_model():
    # dir
    if not os.path.isdir(variable.MODEL_DIR):
        raise Exception('Model directory is missing')
    model_file_list = glob.glob(variable.MODEL_DIR + '*.json')
    model_file_list.sort(reverse=True)
    if len(model_file_list) > 0:
        file_name = os.path.splitext(os.path.basename(model_file_list[0]))[0]
        if os.path.isfile(variable.MODEL_DIR + file_name + '.h5'):
            json_file = open(model_file_list[0], 'r')
            model_json = json_file.read()
            json_file.close()
            model = model_from_json(model_json)
            model.load_weights(variable.MODEL_DIR + file_name + '.h5')
        else:
            raise Exception('Weight is missing')
    else:
        raise Exception('Model is missing')
    return model


def write_model(model):
    # dir
    variable.set_time_stamp()
    if not os.path.isdir(variable.MODEL_DIR):
        os.mkdir(variable.MODEL_DIR)
    model_json = model.to_json()
    with open(variable.get_model_file(), 'w') as json_file:
        json_file.write(model_json)
        json_file.close()
    model.save_weights(variable.get_model_weight())
    pyplot.save_figure()


def save_figure(plt):
    if not os.path.isdir(variable.MODEL_DIR):
        os.mkdir(variable.MODEL_DIR)
    f1 = plt.figure(1)
    f2 = plt.figure(2)
    f1.savefig(variable.get_fig_acc_name())
    f2.savefig(variable.get_fig_loss_name())


def save_variable(opt, tr, wa=None):
    if not os.path.isdir(variable.MODEL_DIR):
        os.mkdir(variable.MODEL_DIR)
    with open(variable.get_var_name(), 'w') as var_file:
        var_file.write('Optimizer: ' + str(opt) + '\n')
        var_file.write('Training rate: ' + str(tr) + '\n')
        if wa is not None:
            var_file.write('Weight amplifier: ' + str(wa) + '\n')
        var_file.close()
