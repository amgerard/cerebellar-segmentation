import numpy as np


def norm_to_zscore(img_array, cer_arr, max_z_score=4.0):
        temp_cer_mask = (cer_arr > 0)
        temp_masked_img = img_array[temp_cer_mask]
        temp_avg_img = np.mean(temp_masked_img)
        temp_std_img = np.std(temp_masked_img)
        img_array[temp_cer_mask] = (img_array[temp_cer_mask] - temp_avg_img) / temp_std_img
        img_array[temp_cer_mask == 0] = 0
        img_array[img_array > max_z_score] = max_z_score
        del temp_masked_img, temp_avg_img, temp_std_img
        return img_array