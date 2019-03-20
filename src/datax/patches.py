import random
import numpy as np
import config

s1, a1 = config.patch_size_subtracter, config.patch_size_adder
def is_in_bounds(im, idx):
    i, j, k = idx
    return \
        i - s1 >= 0 and i + a1 < im.shape[0] and \
        j - s1 >= 0 and j + a1 < im.shape[1] and \
        k - s1 >= 0 and k + a1 < im.shape[2]


def to_vector(lists):
    return [item for sublist in lists for item in sublist]


def is_flip(p):
    return random.random() < p


def get_training_example(im_idx, im_arr, im_truth_arr, lbl, idx, edge_idxs_dict=None):
    if not is_in_bounds(im_arr, idx): return [], []
    x_for_idx = to_vector([[im_idx], idx, [0, 0, 0], [False]])
    y_for_idx = lbl - 1

    '''i, j, k = idx
    if edge_idxs_dict is not None and (i, j, k) in edge_idxs_dict:
        for _ in range(5):
            x_for_lbl.append(
                getX([[im_idx], idx, np.random.uniform(-5, 5, 3), [is_flip(0.5)]]))
            y_for_lbl.append(lbl - 1)'''
    return [x_for_idx], [y_for_idx]


def get_examples_for_label(im_idx, im_arr, im_truth_arr, lbl, edge_idxs_dict=None):
    x_for_lbl, y_for_lbl = [], []
    lbl_idxs = np.where(im_truth_arr == lbl)
    for idx in zip(*lbl_idxs):
        x_for_idx, y_for_idx = get_training_example(im_idx, im_arr, im_truth_arr, lbl, idx, edge_idxs_dict)
        x_for_lbl.extend(x_for_idx)
        y_for_lbl.extend(y_for_idx)

    return x_for_lbl, y_for_lbl


def get_examples(im_idx, im_arr, im_truth_arr, edge_idxs_dict=None):
    x_list, y_list = [], []
    for lbl in range(1, 4):
        x_for_lbl, y_for_lbl = get_examples_for_label(im_idx, im_arr, im_truth_arr, lbl, edge_idxs_dict)
        x_list.extend(x_for_lbl)
        y_list.extend(y_for_lbl)
    
    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    y_arr_one_hot = np.zeros([y_arr.shape[0], 3])
    y_arr_one_hot[np.arange(y_arr.shape[0]), y_arr] = 1
    return x_arr, y_arr_one_hot


if __name__ == '__main__':
    pass
