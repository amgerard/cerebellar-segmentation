import random
import numpy as np
from sklearn.model_selection import train_test_split
from datax.read_and_convert import get_imgs_and_edges
from datax.get_patches_3d import sample_patches

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


def get_patches(x, y, im_idx, edges=None):
    x_arr, y_arr = sample_patches(im_idx, x, y, edges)
    '''x_and_y = [[x_arr, y_arr]]
    random.shuffle(x_and_y)
    x_shuffle = np.concatenate([a[0] for a in x_and_y], axis=0)'''
    x_shuffle, y_shuffle = get_shuffled(x_arr, y_arr)
    print('x patch count: ', x_shuffle.shape)
    #y_shuffle = np.concatenate([a[1] for a in x_and_y], axis=0)
    print('y patch count: ', y_shuffle.shape)
    return x_shuffle, y_shuffle

"""
read images, do canny, return (idxs, rots, flip) and label
:param t1_path: path to t1 training image
:param t2_path: path to t2 training image
:param gt_path: path to ground-truth for t1 and t2
:return: feature vectors with corresponding labels (by voxel)
"""
def get_trn_data(t1_path, t2_path, gt_path, cer_path, do_val=False):
    x, y, edges, t1, t2 = get_imgs_and_edges(t1_path, t2_path, gt_path, cer_path)
    all_imgs.append(t1)
    all_imgs.append(t2)
    all_imgs.append(x)
    im_idx = len(all_imgs) - 1

    if y is None:
        print('NO LABELS FOR BRAIN!')
        return None

    if do_val == False:
        x_trn, y_trn = get_patches(x, y, im_idx, edges)
        x_tst, y_tst = get_patches(x, y, im_idx,  None)
        return x_trn, x_tst, y_trn, y_tst

    y_trn = np.copy(y)
    y_val = np.copy(y)

    mid_axis_2 = int(y.shape[2] / 2)
    y_trn[:, :, :mid_axis_2] = 0
    y_val[:, :, mid_axis_2:] = 0

    x_trn, y_trn = get_patches(x, y_trn, im_idx, edges)
    #x_val, y_val = get_patches(x, y_val, edges)
    x_tst, y_tst = get_patches(x, y_val, im_idx, None)

    return x_trn, x_tst, y_trn, y_tst
    #return x_trn, x_val[:100000, :], y_trn, y_val[:100000]
    # return train_test_split(X_new, y_new, test_size=0.15, random_state=42)


def get_test_data(t1_path, t2_path, cer_path):
    x, y, edges, t1, t2 = get_imgs_and_edges(t1_path, t2_path, cer_path, cer_path)
    all_imgs.append(t1)
    all_imgs.append(t2)
    all_imgs.append(x)
    import predict_mask
    #crbl_map = predict_mask.get_crbl_map(t1_path, cer_path)
    import SimpleITK as sitk
    crbl_img = sitk.ReadImage(cer_path, sitk.sitkUInt16)
    crbl_np = sitk.GetArrayFromImage(crbl_img)
    crbl_idxs = np.where(crbl_np > 0)
    return np.array(zip(*crbl_idxs))


if __name__ == '__main__':
    pass
