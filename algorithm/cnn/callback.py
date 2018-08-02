import keras
from util import file_processing, pyplot


class Callback(keras.callbacks.Callback):
    def __init__(self, has_test=False):
        self.train_line, self.test_line, self.train_t_line, self.test_t_line = None, None, None, None
        self.train_acc, self.test_acc, self.train_t_acc, self.test_t_acc = [], [], [], []
        self.loss_line = None
        self.loss = []
        self.train_data = file_processing.read_data('train')
        self.train_t_data = file_processing.read_data('train', folder_list=['0'])
        self.has_test = has_test
        if has_test:
            self.test_data = file_processing.read_data('test')
            self.test_t_data = file_processing.read_data('test', folder_list=['0'])

    def on_train_begin(self, logs=None):
        self.train_line, self.test_line, self.train_t_line, self.test_t_line = pyplot.initial_acc_figure(self.has_test)
        self.loss_line = pyplot.initial_loss_figure()
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        result = self.model.evaluate(self.train_data['x'], self.train_data['y'],
                                     batch_size=self.params['batch_size'], verbose=0)
        self.loss.append(result[0])
        self.train_acc.append(result[1])
        self.train_t_acc.append(self.model.evaluate(self.train_t_data['x'], self.train_t_data['y'],
                                                    batch_size=self.params['batch_size'], verbose=0)[1])
        pyplot.update_loss_figure(epoch + 1, self.loss, self.loss_line)
        if self.has_test:
            self.test_acc.append(self.model.evaluate(self.test_data['x'], self.test_data['y'], verbose=0)[1])
            self.test_t_acc.append(self.model.evaluate(self.test_t_data['x'], self.test_t_data['y'],
                                                       verbose=0)[1])
            pyplot.update_acc_figure(epoch + 1, self.train_acc, self.train_line, self.train_t_acc, self.train_t_line,
                                     self.test_acc, self.test_line, self.test_t_acc, self.test_t_line)
        else:
            pyplot.update_acc_figure(epoch + 1, self.train_acc, self.train_line, self.train_t_acc, self.train_t_line)
        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return
