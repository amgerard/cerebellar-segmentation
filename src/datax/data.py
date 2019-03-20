import random
import numpy as np
from datax.read_and_convert import get_imgs_and_edges

all_imgs = []  # used across project

def get_shuffled(x_arr, y_arr):
    x_and_y = [np.append(x_arr[i],y_arr[i]) for i in range(x_arr.shape[0])]
    np.random.shuffle(x_and_y)
    x_dim1 = x_arr.shape[1]
    x_shuffle = [a[:x_dim1] for a in x_and_y]
    y_shuffle = [a[x_dim1:] for a in x_and_y]
    x_shuffle = np.array(x_shuffle)
    y_shuffle = np.array(y_shuffle)
    return x_shuffle, y_shuffle


def get_test_data(t1_path, t2_path, cer_path):
    x, y, edges, t1, t2 = get_imgs_and_edges(t1_path, t2_path, cer_path, cer_path)
    all_imgs.append(t1)
    all_imgs.append(t2)
    all_imgs.append(x)
    import SimpleITK as sitk
    crbl_img = sitk.ReadImage(cer_path, sitk.sitkUInt16)
    crbl_np = sitk.GetArrayFromImage(crbl_img)
    crbl_idxs = np.where(crbl_np > 0)
    return np.array(zip(*crbl_idxs))


if __name__ == '__main__':
    pass
