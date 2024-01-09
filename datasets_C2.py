from tensorflow.python.keras.utils.data_utils import get_file
import os
import numpy as np

# from google_drive_downloader import GoogleDriveDownloader as gdd

# credit for https://www.microsoft.com/en-us/download/details.aspx?id=54765

IMG_SIZE = 50


def load_data(path):
    save_data_path = os.path.join(path, 'cat_vs_dog_data.npy')
    save_label_path = os.path.join(path, 'cat_vs_dog_label.npy')
    data_from_numpy = np.load(save_data_path)
    label_from_numpy = np.load(save_label_path)
    data = data_from_numpy.copy()
    label = label_from_numpy.copy()

    '''
    Split Dataset into Train vs Test:  20000 vs 5000
    '''
    test_data = data[-5000:]
    test_label = label[-5000:]
    train_data = data[:-5000]
    train_label = label[:-5000]

    return train_data, train_label, test_data, test_label


def CAT_VS_DOG():
    dirname = 'cat_vs_dog.tar.gz'
    origin = ' https://homepage.divms.uiowa.edu/~zhuoning/datasets/cat_vs_dog.tar.gz'
    path = get_file(
        dirname,
        origin=origin,
        extract=True,
        file_hash=None)

    # remove suffix of filename
    path = path.replace('.tar.gz', '')
    path = path.replace('.zip', '')

    train_X, train_Y, test_X, test_Y = load_data(path)

    # convert data type
    train_X, train_Y = train_X.astype(float), train_Y.astype(np.int32)
    test_X, test_Y = test_X.astype(float), test_Y.astype(np.int32)

    return (train_X, train_Y), (test_X, test_Y)


