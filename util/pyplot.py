from matplotlib import pyplot as plt
from util import file_processing


def initial_acc_figure(has_test=False):
    plt.ion()
    f1 = plt.figure(1)
    train_line, = plt.plot([0], [0], label='train acc')
    train_t_line, = plt.plot([0], [0], label='train tumor acc')
    if has_test:
        test_line, = plt.plot([0], [0], label='test acc')
        test_t_line, = plt.plot([0], [0], label='test tumor acc')
    else:
        test_line, test_t_line = None, None
    axes = f1.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 1])
    axes.set_xlabel('epoch')
    axes.set_ylabel('accuracy')
    plt.legend()
    f1.show()
    plt.pause(0.01)
    return train_line, test_line, train_t_line, test_t_line


def initial_loss_figure():
    plt.ion()
    f2 = plt.figure(2)
    loss_line, = plt.plot([0], [0], label='loss')
    axes = f2.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 10])
    axes.set_xlabel('epoch')
    axes.set_ylabel('loss')
    plt.legend()
    f2.show()
    plt.pause(0.01)
    return loss_line


def update_acc_figure(epoch, train_acc, train_line, train_t_acc, train_t_line,
                      test_acc=None, test_line=None, test_t_acc=None, test_t_line=None):
    f1 = plt.figure(1)
    axes = f1.gca()
    axes.set_xlim([0, epoch + 10])
    train_line.set_xdata(list(range(0, epoch + 1)))
    train_list = list(train_acc)
    train_list.insert(0, 0)
    train_line.set_ydata(train_list)
    train_t_line.set_xdata(list(range(0, epoch + 1)))
    train_t_list = list(train_t_acc)
    train_t_list.insert(0, 0)
    train_t_line.set_ydata(train_t_list)
    if test_acc is not None:
        test_line.set_xdata(list(range(0, epoch + 1)))
        test_list = list(test_acc)
        test_list.insert(0, 0)
        test_line.set_ydata(test_list)
    if test_t_acc is not None:
        test_t_line.set_xdata(list(range(0, epoch + 1)))
        test_t_list = list(test_t_acc)
        test_t_list.insert(0, 0)
        test_t_line.set_ydata(test_t_list)
    plt.draw()
    plt.pause(0.01)


def update_loss_figure(epoch, loss, loss_line):
    f2 = plt.figure(2)
    axes = f2.gca()
    axes.set_xlim([1, epoch + 10])
    axes.set_ylim([0, int(max(loss)) + 1])
    loss_line.set_xdata(list(range(1, epoch + 1)))
    loss_line.set_ydata(loss)
    plt.draw()
    plt.pause(0.01)


def save_figure():
    file_processing.save_figure(plt)

