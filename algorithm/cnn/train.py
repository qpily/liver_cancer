import random

import numpy as np
from keras import losses
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.regularizers import l2

from algorithm.cnn import callback
from util import file_processing, pyplot, variable
from pre_processing import special

EPOCH_NUM = 20
BATCH_NUM = 100


def train(train_data, class_weight=None, has_test=False):
    my_callback = callback.Callback(has_test)

    x = train_data['x']
    y = train_data['y']
    if len(x) > 0 and len(y) > 0:
        model = __model__(x, y)
        model.fit(x, y, batch_size=1024, epochs=EPOCH_NUM, class_weight=class_weight, callbacks=[my_callback])
        scores = model.evaluate(x, y, verbose=0)
        print("%s: %.4f" % (model.metrics_names[0], scores[0]))
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        file_processing.write_model(model)
        if class_weight is not None:
            file_processing.save_variable('nadam', variable.TRAINING_RATE, special.WEIGHT_AMP)
        else:
            file_processing.save_variable('nadam', variable.TRAINING_RATE)


def train_on_batch(train_data, class_weight=None, test_data=None):
    if len(train_data['x']) > 0 and len(train_data['y']) > 0:
        model = __model__(train_data['x'], train_data['y'])
        train_acc, test_acc = [], []
        train_line, test_line = pyplot.initial_acc_figure()
        for epoch in range(1, EPOCH_NUM + 1):
            x_batch = np.array_split(train_data['x'], BATCH_NUM)
            y_batch = np.array_split(train_data['y'], BATCH_NUM)
            mix_batch = list(zip(x_batch, y_batch))
            random.shuffle(mix_batch)
            x_batch, y_batch = zip(*mix_batch)
            for i, (x, y) in enumerate(zip(iter(x_batch), iter(y_batch))):
                train_return = model.train_on_batch(x, y, class_weight=class_weight)
                print('batch: ' + str(i) + '/' + str(BATCH_NUM) + ', accuracy: ' + '%.2f%%\r' % train_return[1])
            print('start evaluate train acc')
            train_acc.append(model.evaluate(train_data['x'], train_data['y'],
                                            batch_size=len(x_batch[0]), verbose=0)[1])
            if test_data is not None:
                print('start evaluate test acc')
                test_acc.append(model.evaluate(test_data['x'], test_data['y'], verbose=0)[1])
                pyplot.update_acc_figure(epoch, train_acc, train_line, test_acc, test_line)
            else:
                pyplot.update_acc_figure(epoch, train_acc, train_line)
        file_processing.write_model(model)


def execute(weight_func=None, on_batch=False, has_test=False):
    data = file_processing.read_data('train')
    class_weight = None
    if weight_func:
        class_weight = weight_func(data['count'])
    if on_batch:
        test_data = None
        if has_test:
            test_data = file_processing.read_data('test')
        train_on_batch(data, class_weight, test_data)
    else:
        train(data, class_weight, has_test)


def __model__(x, y):
    x_shape = x[0].shape
    y_shape = y[0].shape

    model = Sequential()
    # 1
    model.add(Conv2D(64, (3, 3), input_shape=x_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    # 2
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(512, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(y_shape[0], ))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    nadam = Nadam(lr=variable.TRAINING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss=losses.categorical_crossentropy, optimizer=nadam, metrics=['accuracy'])
    return model


