import numpy as np
import SimpleITK as sitk
import itertools
from normalize import normalize, norm_to_zscore

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
    #t1_img[cer_image == 0] = 0
    #t2_img[cer_image == 0] = 0

    #t2_img = sitk.Resample(
    #    sitk.ReadImage(t2_intensity_image, sitk.sitkUInt16),
    #    t1_img,IDTFM,sitk.sitkLinear)

    print('cerebellum mask')
    cer_array = sitk.GetArrayFromImage(sitk.ReadImage(cer_image))

    print('convert t1/t2 to numpy and normalize...')
    ## Start forward features
    #t1_array = img_as_float(sitk.GetArrayFromImage(t1_img))
    #t2_array = img_as_float(sitk.GetArrayFromImage(t2_img))
    #t1_array = norm_to_zscore(sitk.GetArrayFromImage(t1_img), cer_array)
    #t1_array = normalize(sitk.GetArrayFromImage(t1_img))
    #t2_arr = sitk.GetArrayFromImage(t2_img)
    #t2_array = norm_to_zscore(t2_arr, cer_array)
    #t2_array = normalize(t2_arr)

    t1_array = sitk.GetArrayFromImage(t1_img)
    t2_array = sitk.GetArrayFromImage(t2_img)
    t1_array[cer_array == 0] = 0
    t2_array[cer_array == 0] = 0
    t1_array = norm_to_zscore(t1_array, cer_array)
    t2_array = norm_to_zscore(t2_array, cer_array)


    #print('retrieve image from resampled t2')
    #t2_resample = sitk.GetImageFromArray(t2_arr)

    print('concat t1/t2 to single 4D image...')
    t1 = np.expand_dims(t1_array, axis=3)
    t2 = np.expand_dims(t2_array, axis=3)
    image = np.concatenate([t1, t2], axis=3)
    lbls_array = None
    if gt_image != '':
        print('load labels and resample to same t1 voxel lattice...')
        lbls_array = sitk.GetArrayFromImage(sitk.ReadImage(gt_image))
        #if 'brain2' in gt_image or 'brain3' in gt_image:
        #    lbls_array[lbls_array == 2] = 0 # allllllllllllleeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeexxxxxxxxxxxxxxxxxx
        #if 'brain1' in gt_image: # 4/28/18
        #    lbls_array[lbls_array == 1] = 0
        #    lbls_array[lbls_array == 3] = 0

    #lbls_array = sitk.GetArrayFromImage(sitk.Resample(
    #    sitk.ReadImage(gt_image, sitk.sitkInt8),
    #    t1_img,IDTFM,sitk.sitkNearestNeighbor)) #* crbl_map
    #lbls_array = injectToProbLabels(man_lbls,-1,man_lbls==0) * crbl_map
    #lbls_array = man_lbls * sitk.Cast(man_lbls < 4, sitk.sitkInt8)

    # edges (directly on images)
    #print('canny edge detector')
    #t2_edges = get_edge_indexes(t2_resample) # t2_img)
    #edges_dict = merge_dicts(t1_edges, t2_edges)
    edges_dict = {}
    '''edge_nbr_dict = {}
    for key in edges_dict:
        i, j, k = key
        #offsets = itertools.product([-4, -3, -2, -1, 0, 1, 2, 3, 4], repeat=3)
        offsets = itertools.product(range(-6,7), repeat=3)
        for ofst in offsets:
            nbr_idx = (i + ofst[0], j + ofst[1], k + ofst[2])
            if nbr_idx not in edges_dict:
                edge_nbr_dict[nbr_idx] = True
    '''
    print('total edge voxels: ', len(edges_dict))
    #edges_dict = edge_nbr_dict
    #edges_dict = {(i-min0, j-min1, k-min2): True for (i, j, k) in edges_dict.keys()}

    print('total edge neighbor voxels: ', len(edges_dict))

    # stats
    print('image shape: ', image.shape)
    print('t1 min,max,mean: ', t1.min(), t1.max(), t1.mean())
    print('t2 min,max,mean: ', t2.min(), t2.max(), t2.mean())
    if lbls_array is not None:
        print('total labeled voxels: ', len(np.where(lbls_array > 0)[0]))

    return image, lbls_array, None, t1_img, t2_img
    '''for imX,lbl in [(t1_array,'t1'),(t2_array,'t2'),(lbls_array,'lbls')]:
        imXX = sitk.Cast(sitk.GetImageFromArray(imX[min0:max0, min1:max1, min2:max2]), sitk.sitkFloat32)
        print(imXX.GetSize())
        sitk.WriteImage(imXX, lbl + "_crop.nii.gz")'''

if __name__ == "__main__":
    import sys
    import time

    start = time.time()

    t1_intensity_image = sys.argv[1]
    t2_intensity_image = sys.argv[2]

    crbl_mask_image = sys.argv[3]

    get_data5(t1_intensity_image, t2_intensity_image, crbl_mask_image)
    # print("File {0} written.".format(out_file))
    print("done")
    end = time.time()
    print("Elapsed Time: {0}".format(end - start))
