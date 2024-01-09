from tensorflow.python.keras.utils.data_utils import get_file
import os
import numpy as np

# credit from https://github.com/mttk/STL10/blob/c76083f1541ab73b3fa2d820697eeb0e31b01f2b/stl10_input.py#L40


IMG_SIZE = 96


def load_data(path, mode='train'):
    with open(path + '/%s_X.bin' % mode) as f:
        images = np.fromfile(f, dtype=np.uint8, count=-1)
        images = np.reshape(images, (-1, 3, IMG_SIZE, IMG_SIZE))
        images = np.transpose(images, (0, 3, 2, 1))

    with open(path + '/%s_y.bin' % mode) as f:
        labels = np.fromfile(f, dtype=np.uint8, count=-1)
        labels = labels - 1  # class labels are originally in 1-10 format. Convert them to 0-9 format
    return images, labels


def STL10():
    dirname = 'stl10_binary.tar.gz'
    origin = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
    path = get_file(
        dirname,
        origin=origin,
        extract=True,
        file_hash=None)

    # remove suffix of filename
    path = path.replace('.tar.gz', '')
    path = path.replace('.zip', '')

    train_X, train_Y = load_data(path, mode='train')
    test_X, test_Y = load_data(path, mode='test')

    # convert data type
    train_X, train_Y = train_X.astype(float), train_Y.astype(np.int32)
    test_X, test_Y = test_X.astype(float), test_Y.astype(np.int32)

    return (train_X, train_Y), (test_X, test_Y)

