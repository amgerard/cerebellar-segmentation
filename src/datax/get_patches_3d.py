import random
import numpy as np
import config

s1, a1 = config.patch_size_subtracter, config.patch_size_adder
def is_in_bounds(im, i, j, k):
    return \
        i - s1 >= 0 and i + a1 < im.shape[0] and \
        j - s1 >= 0 and j + a1 < im.shape[1] and \
        k - s1 >= 0 and k + a1 < im.shape[2]


def getX(l):
    return [item for sublist in l for item in sublist]


def flip(p):
    return random.random() < p


def sample_patches(im_idx, x_c, y_c, edge_idxs_dict=None):
    '''

    :param im_idx: if only we had multiple training images (unused)
    :param x_c: image as npy
    :param y_c: ground-truth as npy
    :param edge_idxs_dict: dictionary of edge indices
    :return: feature vector (idx,rot,flip) and label
    '''
    # edge_idxs_dict = get_edge_indexes(x_c)
    X_tmp = []
    y_tmp = []
    for lbl in range(1, 4):  # each label
        lbl_idxs = np.where(y_c == lbl)
        for idx in zip(*lbl_idxs):  # each idx for this label
            i, j, k = idx
            if not is_in_bounds(x_c,i,j,k):  # is valid
                #print('patch out of bounds!')
                continue
            # to feature vector
            X_tmp.append(getX([[im_idx], idx, [0, 0, 0], [False]]))
            y_tmp.append(lbl - 1)  # label

            # set random rotate/flip prms if idx is an edge
            if edge_idxs_dict is not None and (i, j, k) in edge_idxs_dict:
                for _ in range(5): # 20
                    X_tmp.append(
                        getX([[im_idx], idx, np.random.uniform(-5, 5, 3), [flip(0.5)]])) # -15, 15
                    y_tmp.append(lbl - 1)
    # list to numpy
    X_new = np.array(X_tmp)
    y = np.array(y_tmp)

    # one-hot
    y_new = np.zeros([y.shape[0], 3])
    y_new[np.arange(y.shape[0]), y] = 1

    return X_new, y_new

def format_training_samples(im_idx, lbl_idxs, y_c):
    X_tmp = []
    y_tmp = []
    if True:
        for idx in zip(*lbl_idxs):  # each idx for this label
            i, j, k = idx
            if not is_in_bounds(y_c,i,j,k):  # is valid
                continue
            X_tmp.append(getX([[im_idx], idx, [0, 0, 0], [False]]))
            y_tmp.append(y_c[i,j,k] - 1)  # label

    # list to numpy
    X_new = np.array(X_tmp)
    y = np.array(y_tmp)

    # one-hot
    y_new = np.zeros([y.shape[0], 3])
    y_new[np.arange(y.shape[0]), y] = 1

    return X_new, y_new

if __name__ == '__main__':
    pass
