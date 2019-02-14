import numpy as np
import SimpleITK as sitk
import itertools
from datax.normalize import norm_to_zscore

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def get_edge_indexes(image): #_np):
	#image = sitk.GetImageFromArray(image_np[:, :, :, 0])
	edges = sitk.CannyEdgeDetection(sitk.Cast(image, sitk.sitkFloat32), lowerThreshold=0.0,
									upperThreshold=200.0, variance=(5.0, 5.0, 5.0))
	edge_indexes = np.where(sitk.GetArrayFromImage(edges) == 1.0)
	print('test', len(edge_indexes[0]))
	edge_indexes_dict = {idx: True for idx in zip(*edge_indexes)}
	return edge_indexes_dict

def get_imgs_and_edges(t1_intensity_image, t2_intensity_image, gt_image, cer_image):
    man_lbls_codes = {"WM": 1, "CSF": 2, "GM": 3, "WM_GUESS":7, "GM_GUESS":9, "CSF_GUESS":8}

    print('load t1/t2 and resample to same t1 voxel lattice...')
    # NOTE RESAMPLE EVERYTHING TO SAME T1 VOXEL LATTICE FOR SIMPLIFYING NUMPY.
    IDTFM=sitk.Transform() # Make an indentity transform
    t1_img = sitk.ReadImage(t1_intensity_image,sitk.sitkFloat32)#, sitk.sitkUInt16)
    t2_img = sitk.ReadImage(t2_intensity_image,sitk.sitkFloat32)#, sitk.sitkUInt16)

    cer_array = sitk.GetArrayFromImage(sitk.ReadImage(cer_image))

    t1_array = sitk.GetArrayFromImage(t1_img)
    t2_array = sitk.GetArrayFromImage(t2_img)
    t1_array[cer_array == 0] = 0
    t2_array[cer_array == 0] = 0
    t1_array = norm_to_zscore(t1_array, cer_array)
    t2_array = norm_to_zscore(t2_array, cer_array)

    print('concat t1/t2 to single 4D image...')
    t1 = np.expand_dims(t1_array, axis=3)
    t2 = np.expand_dims(t2_array, axis=3)
    image = np.concatenate([t1, t2], axis=3)
    lbls_array = None
    if gt_image != '':
        print('load labels and resample to same t1 voxel lattice...')
        lbls_array = sitk.GetArrayFromImage(sitk.ReadImage(gt_image))
    
    return image, lbls_array, None, t1_img, t2_img


if __name__ == "__main__":
    pass

