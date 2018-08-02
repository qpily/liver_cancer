from algorithm.cnn import test, train
from pre_processing import special, data_augmentation, data_processing

# data_processing.data_process(0, [], special.two_type)
# data_augmentation.execute('train')
# data_augmentation.execute('test')
train.execute(special.two_type_weight, False, True)
# train.execute(None, False, True)
# test.execute()
# predict.execute()
