from __future__ import division, print_function, absolute_import
import SimpleITK as sitk
import numpy as np
from datax.normalize import norm_to_zscore
from datax.patches import get_examples

def read_image(image_path):
    img = sitk.ReadImage(image_path, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(img)


class CerebellumMriData(object):

    def __init__(self, t1_path, t2_path, cerebellum_mask_path, ground_truth_path = ""): 
        self.t1_path = t1_path
        self.t2_path = t2_path
        self.cerebellum_mask_path = cerebellum_mask_path
        self.ground_truth_path = ground_truth_path
        self.load()
    
    def load(self):
        t1_arr = read_image(self.t1_path)
        t2_arr = read_image(self.t2_path)
        cerebellum_mask_arr = read_image(self.cerebellum_mask_path)
        if self.ground_truth_path != "":
            self.ground_truth_arr = read_image(self.ground_truth_path)

        t1_arr[cerebellum_mask_arr == 0] = 0
        t2_arr[cerebellum_mask_arr == 0] = 0
        t1_arr = norm_to_zscore(t1_arr, cerebellum_mask_arr)
        t2_arr = norm_to_zscore(t2_arr, cerebellum_mask_arr)

        t1_and_t2 = [np.expand_dims(t1_arr, axis=3), np.expand_dims(t2_arr, axis=3)]
        self.image_arr = np.concatenate(t1_and_t2, axis=3)

    def get_training_data(self, im_idx):
        return get_examples(im_idx, self.image_arr, self.ground_truth_arr)
    
    def get_testing_data(self):
        pass # return x
