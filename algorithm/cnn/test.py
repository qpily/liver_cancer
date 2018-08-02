from util import file_processing, variable
from keras.optimizers import Adam


def test(test_data, model):
    x = test_data['x']
    y = test_data['y']
    if len(x) > 0 and len(y) > 0:
        adam = Adam(lr=variable.TRAINING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        scores = model.evaluate(x, y, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def execute():
    model = file_processing.read_model()
    data = file_processing.read_data('test')
    test(data, model)
