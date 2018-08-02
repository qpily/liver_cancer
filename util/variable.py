from datetime import datetime

RAW_DIR = './raw/'
DATA_DIR = './data/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'
PREDICT_DIR = DATA_DIR + 'predict/'
MODEL_DIR = './model/'

TRAINING_RATE = 0.00012

__time_stamp__ = ''


def set_time_stamp():
    global __time_stamp__
    __time_stamp__ = datetime.now().strftime('%Y_%m_%d_%H_%M')


def get_model_file():
    return MODEL_DIR + 'model_' + __time_stamp__ + '.json'


def get_model_weight():
    return MODEL_DIR + 'model_' + __time_stamp__ + '.h5'


def get_fig_acc_name():
    return MODEL_DIR + 'model_' + __time_stamp__ + '_acc.png'


def get_fig_loss_name():
    return MODEL_DIR + 'model_' + __time_stamp__ + '_loss.png'


def get_var_name():
    return MODEL_DIR + 'model_' + __time_stamp__ + '.var'