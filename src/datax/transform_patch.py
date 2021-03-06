import SimpleITK as sitk
import numpy as np
import config

s1, a1 = config.patch_size_subtracter, config.patch_size_adder# 16,17
p_sz = config.patch_size


def is_in_bounds(im, i, j, k):
    return i - s1 >= 0 and i + a1 < im.shape[0] and j - s1 >= 0 and j + \
        a1 < im.shape[1] and k >= 0 and k < im.shape[2]

def extract_patch(image_arr, patch_idx):
    i, j, k = [int(x) for x in patch_idx]

    if config.patch_num_dim == 3:
        im_shape = image_arr.shape
        imin,jmin,kmin = max(i-s1,0),max(j-s1,0),max(k-s1,0)
        imax,jmax,kmax = min(i+a1,im_shape[0]),min(j+a1,im_shape[1]),min(k+a1,im_shape[2])
        patch = image_arr[imin:imax,jmin:jmax,kmin:kmax,:]
        if patch.shape == (p_sz,p_sz,p_sz,2):
            return patch
        patch_zeros = np.zeros((p_sz,p_sz,p_sz,2))
        d1,d2,d3,d4 = np.indices(patch.shape)
        patch_zeros[d1,d2,d3,d4] = patch
        return patch_zeros

    return image_arr[i - s1:i + a1, j - s1:j + a1, k, :]

def rotate_combine_normalize(image_arr, patch_idx, rotation, do_flip, patch_size=15):
    i, j, k = [int(x) for x in patch_idx]
    if config.patch_num_dim == 3:
        return image_arr[i-s1:i+a1, j-s1:j+a1, k-s1:k+a1, :]

    return image_arr[i - s1:i + a1, j - s1:j + a1, k, :]


def rotate_3d_patch(image, patch_idx, rotation):
    rotation_x, rotation_y, rotation_z = [np.radians(r) for r in rotation]
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(image)
    resample.SetInterpolator(sitk.sitkLinear)

    # Rotate around the physical center of the image.
    rotation_center = image.TransformContinuousIndexToPhysicalPoint(
        [(index - 1) / 2.0 for index in image.GetSize()])
    transform = sitk.Euler2DTransform(
        rotation_center, rotation_x, (0, 0, 0))
    resample.SetTransform(transform)
    rotated_image = resample.Execute(image)
    return rotated_image
