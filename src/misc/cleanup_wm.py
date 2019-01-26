import SimpleITK as sitk
import random

base_dir = "/home/amgerard/src/CerebellumChallenge/CerebellumData/CrblData/1mm/"


def get_brain_paths(brain):
    brain_dir = base_dir
    label_path = brain_dir + brain + '_lbls_1mm.nrrd'
    return [brain_dir + brain + "_t1_1mm.nrrd", brain_dir + brain + "_t2_1mm.nrrd", label_path]


def get_hans_paths(brain):
    base_dir = "/home/amgerard/src/CerebellumChallenge/CerebellumData/CrblData/1mm/hans/"
    brain_dir = base_dir + brain + "/"
    label_path = brain_dir + 'crbl_seg.nii.gz'
    return [brain_dir + "t1_average_BRAINSABC.nii.gz", brain_dir + "t2_average_BRAINSABC.nii.gz", label_path]


def is_near_gm_x(im, i, j, k):
    for a in range(i-1,i+1):
        for b in range(j-1,j+1):
            for c in range(k-1,k+1):
                if im[a,b,c] == 3:
                    return random.randint(0, 3) > 0
    return random.randint(0,7) == 0


def is_near_gm(im, i, j, k):
    sz = im.GetSize()
    if i > 7 and j > 7 and k > 7:
        if i < sz[0]-7 and j < sz[1]-7 and k < sz[2]-7:
            if not is_near_gm_x(im, i, j, k):
                return False
    return True


def cleanup_wm(brain):
    _, _, lbl_path = get_brain_paths(brain)
    lbl_im = sitk.ReadImage(lbl_path)
    sz = lbl_im.GetSize()
    for i in range(sz[0]):
        for j in range(sz[1]):
            for k in range(sz[2]):
                if lbl_im[i,j,k] == 1 and not is_near_gm(lbl_im,i,j,k):
                    lbl_im[i,j,k] = 0
    return lbl_im


def smooth_brain(brain):
    paths = get_brain_paths(brain)
    for path in paths[0:2]:
        im = sitk.ReadImage(path)
        smooth_im = sitk.SmoothingRecursiveGaussian(im, 0.6)
        smooth_im = sitk.Cast(smooth_im, im.GetPixelIDValue())
        sitk.WriteImage(smooth_im, path.replace(".nrrd", "_smooth_p6.nrrd"))

def plot_hist(im_path, cer_path, title='title', bins='auto'):
    import matplotlib.pyplot as plt
    from normalize import norm_to_zscore
    im_arr = sitk.GetArrayFromImage(sitk.ReadImage(im_path, sitk.sitkFloat32))
    cer_arr = sitk.GetArrayFromImage(sitk.ReadImage(cer_path))
    norm_arr = norm_to_zscore(im_arr, cer_arr)
    plt.hist(norm_arr[norm_arr != 0].flatten(), bins=bins)
    plt.title(title)
    plt.show()

if __name__ == '__main__':

    brains = ['brain1', 'brain2', 'brain3', 'brain4', 'brain5']

    #from predict_mask import makeCrblMaps
    #makeCrblMaps('/home/amgerard/src/CerebellumChallenge/CerebellumData/CrblData/1mm/hans/53657/wholeBrain_label.nii.gz')

    #for brain in brains:
    #    smooth_brain(brain)
    #    #break

    #paths = get_brain_paths('brain1')
    #paths = get_hans_paths('55648')
    #plot_hist(paths[0], paths[2], 'brain1 t1 norm')

    for brain in brains:
        im = cleanup_wm(brain)
        sitk.WriteImage(im, base_dir + brain + "_lbls_1mm_cleanx.nrrd")
