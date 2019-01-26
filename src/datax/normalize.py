import os
from glob import glob
import numpy as np
from skimage.util import img_as_float # should be using?
import SimpleITK as sitk

def normalize(slice):
        '''
        INPUT:  (1) a single slice of any given modality (excluding gt)
        (2) index of modality assoc with slice (0=flair, 1=t1, 2=t1c, 3=t2)2
        OUTPUT: normalized slice
        '''
        b, t = np.percentile(slice, (1.0,99.0))
        slice = np.clip(slice, b, t)
        if np.std(slice) == 0:
                return slice
        else:
                return (slice - np.mean(slice)) / np.std(slice)


def create_tensors_from_mha(path):
        t1,t1c,t2,flr,lbl = ('','','','','')
        for p in [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.mha'))]:
                if 'OT' in p: lbl = p
                if 'T1' in p and 'n4corr' in p: t1 = p
                if 'T1c' in p and 'n4corr' in p: t1c = p
                if 'T2' in p: t2 = p
                if 'Flair' in p: flr = p
        img_lbl = sitk.GetArrayFromImage(sitk.ReadImage(lbl))
        imgs = [np.expand_dims(normalize(sitk.GetArrayFromImage(sitk.ReadImage(p))), axis=3) for p in [t1,t1c,t2,flr]]
        img_concat = np.concatenate(imgs, axis=3)
        return img_concat, img_lbl


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