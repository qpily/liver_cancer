import numpy as np

y_min = 0
y_max = 255


def window_transfer(bw, c, w):
    max_w = (c - 0.5 + (w - 1) / 2)
    min_w = (c - 0.5 - (w - 1) / 2)
    for i in range(len(bw)):
        if bw[i] > max_w:
            bw[i] = 1
        elif bw[i] < min_w:
            bw[i] = 0
        else:
            bw[i] = ((bw[i] - (c - 0.5)) / (w - 1) + 0.5)
    return bw


def normalization(bw):
    bw_np = np.asarray(bw)
    bw_zero_center = bw_np - np.mean(bw)
    bw_new = bw_zero_center / np.std(bw_zero_center)
    return bw_new.tolist()


def structure_transfer(json, name_list):
    image_list = []
    for name in name_list:
        window = json[name + '-WINDOW']
        c = window[0]
        w = window[1]
        image = json[name]
        # image = window_transfer(image, c, w)
        image = normalization(image)
        image_list.append(reshape(image))

    image_list = np.asarray(image_list, dtype=np.float32).swapaxes(0, 1).swapaxes(1, 2)
    return image_list


def reshape(image):
    length = int(len(image) ** 0.5)
    if length ** 2 != int(len(image)):
        raise Exception('Input shape is not a square')
    return np.reshape(image, (-1, length))


def rotate(image, k):
    return np.rot90(image, k)


def flip(image, axis):
    return np.flip(image, axis)
