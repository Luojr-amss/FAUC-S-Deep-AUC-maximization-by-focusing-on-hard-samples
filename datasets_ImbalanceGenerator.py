'''
Currently only support for binary classification
'''
import numpy as np


def check_imbalance_binary(Y):
    # numpy array
    num_samples = len(Y)
    pos_count = np.count_nonzero(Y == 1)
    neg_count = np.count_nonzero(Y == 0)
    pos_ratio = pos_count / (pos_count + neg_count)
    print('#SAMPLES: [%d], POS:NEG: [%d : %d], POS RATIO: %.4f' % (num_samples, pos_count, neg_count, pos_ratio))


def ImbalanceGenerator(X, Y, imratio=0.5, shuffle=True, is_balanced=False, random_seed=123):
    '''
    Imbalanced Data Generator
    Reference: https://arxiv.org/abs/2012.03173
    '''

    assert isinstance(X, (np.ndarray, np.generic)), 'data needs to be numpy type!'
    assert isinstance(Y, (np.ndarray, np.generic)), 'data needs to be numpy type!'

    num_classes = np.unique(Y).size

    if num_classes == 2:
        split_index = 0
    elif num_classes == 10:
        split_index = 4
    elif num_classes == 100:
        split_index = 49
    elif num_classes == 1000:
        split_index = 499
    else:
        raise ValueError('TBD!')

    # shuffle before preprocessing (add randomness for removed samples)
    id_list = list(range(X.shape[0]))
    np.random.seed(random_seed)
    np.random.shuffle(id_list)
    X = X[id_list]
    Y = Y[id_list]
    X_copy = X.copy()
    Y_copy = Y.copy()
    Y_copy[Y_copy <= split_index] = 0  # [0, ....]
    Y_copy[Y_copy >= split_index + 1] = 1  # [0, ....]

    if is_balanced == False:
        num_neg = np.where(Y_copy == 0)[0].shape[0]
        num_pos = np.where(Y_copy == 1)[0].shape[0]
        keep_num_pos = int((imratio / (1 - imratio)) * num_neg)
        neg_id_list = np.where(Y_copy == 0)[0]
        pos_id_list = np.where(Y_copy == 1)[0][:keep_num_pos]
        X_copy = X_copy[neg_id_list.tolist() + pos_id_list.tolist()]
        Y_copy = Y_copy[neg_id_list.tolist() + pos_id_list.tolist()]
        # Y_copy[Y_copy==0] = 0

    if shuffle:
        # do shuffle in case batch prediction error
        id_list = list(range(X_copy.shape[0]))
        np.random.seed(random_seed)
        np.random.shuffle(id_list)
        X_copy = X_copy[id_list]
        Y_copy = Y_copy[id_list]

    num_samples = len(X_copy)
    pos_count = np.count_nonzero(Y_copy == 1)
    neg_count = np.count_nonzero(Y_copy == 0)
    pos_ratio = pos_count / (pos_count + neg_count)
    print('NUM_SAMPLES: [%d], POS:NEG: [%d : %d], POS_RATIO: %.4f' % (num_samples, pos_count, neg_count, pos_ratio))

    return X_copy, Y_copy.reshape(-1, 1).astype(float)


