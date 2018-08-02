from util import file_processing
import numpy as np


def predict(x, model):
    if len(x) > 0:
        prediction = model.predict(x)
        print(np.argmax(prediction, axis=1))


def execute():
    model = file_processing.read_model()
    data = file_processing.read_data('predict')
    predict(data['x'], model)
