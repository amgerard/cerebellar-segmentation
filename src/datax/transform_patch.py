import SimpleITK as sitk
import numpy as np
import datax.data as data
from datax.normalize import normalize
import config

s1, a1 = config.patch_size_subtracter, config.patch_size_adder# 16,17
p_sz = config.patch_size

'''
def rotate_combine_normalize(images, patch_idx, rotation, do_flip, patch_size=15):
    i, j, k = [int(x) for x in patch_idx]
    s1, a1 = 7, 8  # 16,17

    t1_img, t2_img = [im[i-15:i+16, j-15:j+16, k-15:k+16] for im in images[:2]]

    t1_rot = rotate_3d_patch(t1_img, patch_idx, rotation)
    t2_rot = rotate_3d_patch(t2_img, patch_idx, rotation)

    t1_array = normalize(sitk.GetArrayFromImage(t1_rot))
    t2_array = normalize(sitk.GetArrayFromImage(t2_rot))

    t1 = np.expand_dims(t1_array, axis=3)
    t2 = np.expand_dims(t2_array, axis=3)
    image = np.concatenate([t1, t2], axis=3)

    i, j, k = [15,15,15]
    patch = image[i - s1:i + a1, j - s1:j + a1, k - s1:k + a1]
    return np.flip(patch, 2) if do_flip else patch
'''
def is_in_bounds(im, i, j, k):
    return i - s1 >= 0 and i + a1 < im.shape[0] and j - s1 >= 0 and j + \
        a1 < im.shape[1] and k >= 0 and k < im.shape[2]

def extract_patch(im_idx, patch_idx):
    '''images = data.all_imgs
    i, j, k = [int(x) for x in patch_idx]
    s1, a1 = 12, 13  # 16,17
    img_full = data.all_imgs[int(im_idx)]
    #if is_in_bounds(img_full, i, j, k) == False: return None
    patch = img_full[i - s1:i + a1, j - s1:j + a1, k, :]'''

    images = data.all_imgs
    i, j, k = [int(x) for x in patch_idx]

    img_full = data.all_imgs[int(im_idx)]
    '''img_crop = img_full[i - 20:i + 21, j - 20:j + 21, k, :]

    t1_img = sitk.GetImageFromArray(img_crop[:, :, 0])
    t2_img = sitk.GetImageFromArray(img_crop[:, :, 1])

    rotation = [0,0,0]
    t1_rot = rotate_3d_patch(t1_img, patch_idx, rotation)
    t2_rot = rotate_3d_patch(t2_img, patch_idx, rotation)

    t1_array = sitk.GetArrayFromImage(t1_rot)
    t2_array = sitk.GetArrayFromImage(t2_rot)
    t1 = np.expand_dims(t1_array, axis=2)
    t2 = np.expand_dims(t2_array, axis=2)
    image = np.concatenate([t1, t2], axis=2)

    i, j, k = [20, 20, 20]
    patch = image[i - s1:i + a1, k - s1:k + a1]'''

    if config.patch_num_dim == 3:
        im_shape = img_full.shape
        imin,jmin,kmin = max(i-s1,0),max(j-s1,0),max(k-s1,0)
        imax,jmax,kmax = min(i+a1,im_shape[0]),min(j+a1,im_shape[1]),min(k+a1,im_shape[2])
        patch = img_full[imin:imax,jmin:jmax,kmin:kmax,:]
        if patch.shape == (p_sz,p_sz,p_sz,2):
            return patch
        patch_zeros = np.zeros((p_sz,p_sz,p_sz,2))
        d1,d2,d3,d4 = np.indices(patch.shape)
        patch_zeros[d1,d2,d3,d4] = patch
        return patch_zeros
        #return img_full[i-s1:i+a1, j-s1:j+a1, k-s1:k+a1, :]

    return img_full[i - s1:i + a1, j - s1:j + a1, k, :]

def rotate_combine_normalize(im_idx, patch_idx, rotation, do_flip, patch_size=15):
    images = data.all_imgs
    i, j, k = [int(x) for x in patch_idx]

    img_full = data.all_imgs[int(im_idx)]
    '''img_crop = img_full[i - 20:i + 21, j - 20:j + 21, k, :]

    #t1_img = sitk.GetImageFromArray(img_crop[:, :, :, 0])
    #t2_img = sitk.GetImageFromArray(img_crop[:, :, :, 1])
    t1_img = sitk.GetImageFromArray(img_crop[:, :, 0])
    t2_img = sitk.GetImageFromArray(img_crop[:, :, 1])

    t1_rot = rotate_3d_patch(t1_img, patch_idx, rotation)
    t2_rot = rotate_3d_patch(t2_img, patch_idx, rotation)

    t1_array = sitk.GetArrayFromImage(t1_rot)
    t2_array = sitk.GetArrayFromImage(t2_rot)

    #t1_array = np.flip(t1_array, 2) if do_flip else t1_array
    #t2_array = np.flip(t2_array, 2) if do_flip else t2_array

    #t1 = np.expand_dims(t1_array, axis=3)
    #t2 = np.expand_dims(t2_array, axis=3)
    #image = np.concatenate([t1, t2], axis=3)
    t1 = np.expand_dims(t1_array, axis=2)
    t2 = np.expand_dims(t2_array, axis=2)
    image = np.concatenate([t1, t2], axis=2)

    i, j, k = [20, 20, 20]
    patch = image[i - s1:i + a1, k - s1:k + a1]
    #patch = image[i - s1:i + a1, j - s1:j + a1, k - s1:k + a1]
    return patch'''
    if config.patch_num_dim == 3:
        return img_full[i-s1:i+a1, j-s1:j+a1, k-s1:k+a1, :]

    return img_full[i - s1:i + a1, j - s1:j + a1, k, :]


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
