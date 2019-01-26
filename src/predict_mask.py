import os
import multiprocessing as mp
import SimpleITK as sitk
import numpy as np

my_cpu_count = mp.cpu_count()

## from myshow import myshow

glb_cached_objects = "/Shared/johnsonhj/HDNI/20170425_CerebellumSegmentation/EditorHelperML/CACHE"
glb_windowSize = 3
glb_radiusL = glb_windowSize // 2;  ## Lowerbound
glb_radiusU = glb_radiusL + 1;  ## Upperbound
glb_img_feature_cnt = pow((glb_radiusL + glb_radiusU), 3)  # 27 neighbors


def display_overlay_callback(t1_image, t2_image, label, z, op):
    """Display label map overlay on image

    Args:
      t1_image (sitk.Image): Input t1 image
      t2_image (sitk.Image): Input t2 image
      label (sitk.Image): Label map image
      z (int): Image slice to display
      op (float): Opacity of overlay

    """
    img_slices = [t1_image[:, :, z], t2_image[:, :, z]]
    lbl_slices = [label[:, :, z], label[:, :, z]]
    img = sitk.Tile(img_slices, [2, 1])
    lbl = sitk.Tile(lbl_slices, [2, 1])
    overlay = sitk.LabelOverlay(img, lbl, op)
    myshow(overlay, dpi=30)


def iminfo(im):
    """Silly function for displaying image information
    :param im: A simpleITK image 
    :return: nothing
    """
    mm = sitk.MinimumMaximumImageFilter()
    mm.Execute(im)
    print(mm.GetMaximum())
    print(mm.GetMinimum())

def injectToProbLabels(lblmap, labelnum, replace_mask):
    """
    Inject the non-zero pixels in replace_mask into lblmap wiht value labelnum
    :param lblmap:  original label map
    :param labelnum: insertion label number
    :param replace_mask:  locations to insert label number
    :return:  new label map with injected values
    """
    keep_mask = sitk.Cast(1 - replace_mask, sitk.sitkInt8)
    replace_mask = sitk.Cast(replace_mask,sitk.sitkInt8)
    lblmap = sitk.Cast(keep_mask * lblmap + replace_mask * labelnum, sitk.sitkInt8)
    return lblmap


def ExtractSubrange(class_index, curr_classes, curr_features, man_lbls_array, max_index, min_index, unrolled_features):
    start = time.time()
    print("Starting:  {0} {1}".format(min_index,max_index))
    for count in range(min_index, max_index):
        idx = class_index[count]
        temp_array = unrolled_features[idx[0], idx[1], idx[2], :]
        curr_features[count:] = temp_array
        curr_classes[count] = man_lbls_array[idx[0], idx[1], idx[2]]
        count += 1
    end = time.time()
    print("Range {0} {1} Done in {2}".format(min_index,max_index, end - start))


def cnvrToProbMap(crbl_flat_idx, ref_img, input_probs, name):
    prob_img = sitk.Cast(sitk.Image(ref_img) * 0,sitk.sitkFloat64)
    prob_img_array = sitk.GetArrayFromImage(prob_img)
    prob_img_fm = prob_img_array.flatten()
    prob_img_fm[crbl_flat_idx] = input_probs.flatten() * 100.0
    prob_img_fm2 = prob_img_fm.reshape(prob_img_array.shape)
    final_prob = sitk.GetImageFromArray(prob_img_fm2)
    final_prob.CopyInformation(prob_img)
    sitk.WriteImage(sitk.Cast(final_prob, sitk.sitkFloat32), name)
    return final_prob


def makeCrblMaps(jf_path):
    """
    Select the label numbers that correspond to cerebellar regions
    :param jf_map: A joint-fusion based label map with many segmentation labels
    :return: A binary label map
    """

    jf_map = sitk.ReadImage(jf_path, sitk.sitkUInt16)
    #iminfo(jf_map)
    crbl_map = (jf_map == 15071) + (jf_map == 15072) + (jf_map == 15073) + (jf_map == 46) + (jf_map == 7) + (jf_map == 47) + (jf_map == 8)
    CSF_LABEL=24
    crbl_map_surface_CSF = ( \
                        (sitk.Cast(sitk.DilateObjectMorphology(crbl_map,3), sitk.sitkUInt16) * jf_map \
                     ) == CSF_LABEL ) \
                     - ( sitk.Cast(crbl_map,sitk.sitkUInt16)*jf_map == CSF_LABEL )
    crbl_map = sitk.Cast( ( crbl_map + crbl_map_surface_CSF > 0 ), sitk.sitkInt8)

    sitk.WriteImage(crbl_map,"crbl_seg.nii.gz")

    #iminfo(crbl_map)
    return crbl_map

def get_crbl_map(t1_intensity_image, crbl_mask_image):
    man_lbls_codes = {"WM": 1, "CSF": 2, "GM": 3, "WM_GUESS":7, "GM_GUESS":9, "CSF_GUESS":8}

    # NOTE RESAMPLE EVERYTHING TO SAME T1 VOXEL LATTICE FOR SIMPLIFYING NUMPY.
    IDTFM=sitk.Transform() # Make an indentity transform
    t1_img = sitk.ReadImage(t1_intensity_image, sitk.sitkUInt16)

    jointFusionName = os.path.realpath(crbl_mask_image)
    jf_map = sitk.Resample(
        sitk.ReadImage(jointFusionName, sitk.sitkUInt16),
        t1_img, IDTFM, sitk.sitkNearestNeighbor)

    #return sitk.GetArrayFromImage(jf_map)
    crbl_map = makeCrblMaps(jf_map)
    return sitk.GetArrayFromImage(sitk.ReadImage('crbl_seg.nii.gz')) #crbl_map)

def save_mask(t1_img,testX,predY,out_file):
    sz = t1_img.GetSize()
    '''for i,idx in enumerate(testX):
        try:
            # t1_img[idx[0],idx[1],idx[2]] = predY[i]+1
            t1_img.SetPixel(idx[0],idx[1],idx[2],predY[i]+1)
        except:
            print('invalid index: ', idx)'''

    t1_arr = sitk.GetArrayFromImage(t1_img) * 0
    #t1_arr[testX] = predY + 1
    #t1_arr = np.zeros(sz)

    for i, idx in enumerate(testX):
        try:
            if predY[i] >= 0:
                t1_arr[idx[0],idx[1],idx[2]] = predY[i]
            #t1_arr.SetPixel(idx[0], idx[1], idx[2], predY[i] + 1)
        except:
            print('invalid index: ', idx)

    tmp = sitk.GetImageFromArray(t1_arr)
    tmp.CopyInformation(t1_img)
    print("Saving file {0}".format(out_file),t1_arr.max(),t1_arr.min())
    sitk.WriteImage(tmp, out_file)
    #sitk.WriteImage(t1_img, out_file)

def save_mask_old(t1_intensity_image,crbl_mask_image,crbl_samples_image,testX,predY):
    man_lbls_codes = {"WM": 1, "CSF": 2, "GM": 3, "WM_GUESS":7, "GM_GUESS":9, "CSF_GUESS":8}

    # NOTE RESAMPLE EVERYTHING TO SAME T1 VOXEL LATTICE FOR SIMPLIFYING NUMPY.
    IDTFM=sitk.Transform() # Make an indentity transform
    t1_img = sitk.ReadImage(t1_intensity_image, sitk.sitkUInt16)

    jointFusionName = os.path.realpath(crbl_mask_image)
    jf_map = sitk.Resample(
        sitk.ReadImage(jointFusionName, sitk.sitkUInt16),
        t1_img, IDTFM, sitk.sitkNearestNeighbor)

    crbl_map = makeCrblMaps(jf_map)

    man_lbls = sitk.Resample(
        sitk.ReadImage(crbl_samples_image, sitk.sitkInt8),
        t1_img,IDTFM,sitk.sitkNearestNeighbor) * crbl_map
    man_lbls = injectToProbLabels(man_lbls,-1,man_lbls==0) * crbl_map
    man_lbls = man_lbls * sitk.Cast(man_lbls < 4, sitk.sitkInt8)

    lbls_array = sitk.GetArrayFromImage(man_lbls)

    ## Start forward features
    t1_array = sitk.GetArrayFromImage(t1_img)

    man_lbls_array = sitk.GetArrayFromImage(man_lbls)
    ## end forward features

    print("Starting Prediction Phase")
    classifier = "CNN"

    y_values = predY
    useAlternateCodes=False
    if useAlternateCodes:
        y_values[y_values == man_lbls_codes["WM"]]  = man_lbls_codes["WM_GUESS"]
        y_values[y_values == man_lbls_codes["CSF"]] = man_lbls_codes["CSF_GUESS"]
        y_values[y_values == man_lbls_codes["GM"]]  = man_lbls_codes["GM_GUESS"]
    else:
        y_values[y_values == man_lbls_codes["WM"]]  = man_lbls_codes["WM"]
        y_values[y_values == man_lbls_codes["CSF"]] = man_lbls_codes["CSF"]
        y_values[y_values == man_lbls_codes["GM"]]  = man_lbls_codes["GM"]

    estim_lbls = sitk.Image(man_lbls) * 0 + 4
    estim_array = sitk.GetArrayFromImage(estim_lbls)
    ef = estim_array.flatten()
    crbl_flat_idx = np.where(sitk.GetArrayFromImage(crbl_map).flatten() == 1)
    ef[crbl_flat_idx] = y_values

    mf = sitk.GetArrayFromImage(man_lbls).flatten()
    ef[mf == 1] = 1
    ef[mf == 2] = 2
    ef[mf == 3] = 3

    ef2 = ef.reshape(estim_array.shape)

    im = sitk.GetImageFromArray(ef2)
    im.CopyInformation(estim_lbls)

    im = sitk.Cast(im, sitk.sitkInt8)

    final_lbl_map = sitk.Cast(im * sitk.Cast(crbl_map, sitk.sitkInt8), sitk.sitkInt8)
    # final_lbl_map[29,160,79] = 6
    out_file = os.path.join("alex_auto_seg3.nrrd")
    print("Saving file {0}".format(out_file))
    sitk.WriteImage(final_lbl_map, out_file)

def updateMask(t1_intensity_image, t2_intensity_image, feature_image,
               crbl_samples_image, crbl_mask_image,
               base_output_dir, output_prefix):
    """
    A function that uses the intensity images and feature images (t1,t2,feature_image
    :param t1_intensity_image:  A t1 weighted image
    :param t2_intensity_image: A t2 weighted image
    :param feature_image:  A hessian image or a gradient magnitude image or similar
    :param crbl_samples_image: 
    :param crbl_mask_image: 
    :param base_output_dir: 
    :param output_prefix: 
    :return: 
    """


    man_lbls_codes = {"WM": 1, "CSF": 2, "GM": 3, "WM_GUESS":7, "GM_GUESS":9, "CSF_GUESS":8}

    # NOTE RESAMPLE EVERYTHING TO SAME T1 VOXEL LATTICE FOR SIMPLIFYING NUMPY.
    IDTFM=sitk.Transform() # Make an indentity transform
    t1_img = sitk.ReadImage(t1_intensity_image, sitk.sitkUInt16)
    t2_img = sitk.Resample(
        sitk.ReadImage(t2_intensity_image, sitk.sitkUInt16),
        t1_img,IDTFM,sitk.sitkLinear)

    feat_img = sitk.Resample(
        sitk.ReadImage(feature_image, sitk.sitkUInt16),
        t1_img,IDTFM,sitk.sitkNearestNeighbor)

    jointFusionName = os.path.realpath(crbl_mask_image)
    jf_map = sitk.Resample(
        sitk.ReadImage(jointFusionName, sitk.sitkUInt16),
        t1_img, IDTFM, sitk.sitkNearestNeighbor)

    crbl_map = makeCrblMaps(jf_map)

    man_lbls = sitk.Resample(
        sitk.ReadImage(crbl_samples_image, sitk.sitkInt8),
        t1_img,IDTFM,sitk.sitkNearestNeighbor) * crbl_map
    man_lbls = injectToProbLabels(man_lbls,-1,man_lbls==0) * crbl_map
    man_lbls = man_lbls * sitk.Cast(man_lbls < 4, sitk.sitkInt8)

    lbls_array = sitk.GetArrayFromImage(man_lbls)

    ## Start forward features
    t1_array = sitk.GetArrayFromImage(t1_img)
    t2_array = sitk.GetArrayFromImage(t2_img)
    feat_img_array = sitk.GetArrayFromImage(feat_img)

    man_lbls_array = sitk.GetArrayFromImage(man_lbls)
    ## end forward features

    crbl_idxs = np.where(sitk.GetArrayFromImage(crbl_map) == 1)

    print("Starting Prediction Phase")
    classifier = "CNN"


    '''y_pred = clf_model.predict(all_features_scaled)

    y_pred_wm = clf_model.predict(wm_features_scaled)
    y_pred_gm = clf_model.predict(gm_features_scaled)
    y_pred_csf = clf_model.predict(csf_features_scaled)

    #crbl_features_scaled = scaler.transform(crbl_features)
    crbl_features_scaled = crbl_features
    del crbl_features
    y_values = clf_model.predict(crbl_features_scaled)


    def print_status(classifier,region, total,wrong):
        print("{0} {1} Number of mislabeled points {2} of a total {3} : {4}% wrong".format(
            classifier,region,wrong,total, wrong/total))

    print("WM IN PRED {0}".format((y_pred == man_lbls_codes["WM"]).sum()))
    print("CSF IN PRED {0}".format((y_pred == man_lbls_codes["CSF"]).sum()))
    print("GM IN PRED {0}".format((y_pred == man_lbls_codes["GM"]).sum()))

    print("WM IN ALL {0}".format((all_classes == man_lbls_codes["WM"]).sum()))
    print("CSF IN ALL {0}".format((all_classes == man_lbls_codes["CSF"]).sum()))
    print("GM IN ALL {0}".format((all_classes == man_lbls_codes["GM"]).sum()))

    print_status(classifier,"All",all_features_scaled.shape[0],(all_classes != y_pred).sum())
    print_status(classifier,"WM",wm_features_scaled.shape[0],(wm_classes != y_pred_wm).sum())
    print_status(classifier,"GM",gm_features_scaled.shape[0],(gm_classes != y_pred_gm).sum())
    print_status(classifier,"CSF",csf_features_scaled.shape[0],(csf_classes != y_pred_csf).sum())

    y_prob = clf_model.predict_proba(csf_features_scaled)

    print(y_prob.shape)
    print(y_values.shape)
    print(np.max(y_prob[:, 0]))
    print(np.max(y_prob[:, 1]))
    print(np.max(y_prob[:, 2]))

    wm_prob = cnvrToProbMap(crbl_flat_idx, man_lbls, y_prob[:, 0],
                            os.path.join(base_output_dir, output_prefix + "_wm_img.nii.gz"))
    gm_prob = cnvrToProbMap(crbl_flat_idx, man_lbls, y_prob[:, 2],
                            os.path.join(base_output_dir, output_prefix + "_gm_img.nii.gz"))
    csf_prob = cnvrToProbMap(crbl_flat_idx, man_lbls, y_prob[:, 1],
                             os.path.join(base_output_dir, output_prefix + "_csf_img.nii.gz"))

    useAlternateCodes=False
    if useAlternateCodes:
        y_values[y_values == man_lbls_codes["WM"]]  = man_lbls_codes["WM_GUESS"]
        y_values[y_values == man_lbls_codes["CSF"]] = man_lbls_codes["CSF_GUESS"]
        y_values[y_values == man_lbls_codes["GM"]]  = man_lbls_codes["GM_GUESS"]
    else:
        y_values[y_values == man_lbls_codes["WM"]]  = man_lbls_codes["WM"]
        y_values[y_values == man_lbls_codes["CSF"]] = man_lbls_codes["CSF"]
        y_values[y_values == man_lbls_codes["GM"]]  = man_lbls_codes["GM"]

    estim_lbls = sitk.Image(man_lbls) * 0 + 4
    estim_array = sitk.GetArrayFromImage(estim_lbls)
    ef = estim_array.flatten()
    ef[crbl_flat_idx] = y_values

    mf = sitk.GetArrayFromImage(man_lbls).flatten()
    ef[mf == 1] = 1
    ef[mf == 2] = 2
    ef[mf == 3] = 3

    ef2 = ef.reshape(estim_array.shape)'''

    im = sitk.GetImageFromArray(ef2)
    im.CopyInformation(estim_lbls)

    im = sitk.Cast(im, sitk.sitkInt8)

    """ HACKS THAT DON"T WORK YET
    if 0 == 1:
        im = injectToProbLabels(im, 1, wm_prob  > 99)
        im = injectToProbLabels(im, 2, csf_prob > 99)
        im = injectToProbLabels(im, 3, gm_prob  > 99)

    if 0 == 1:
        if 1 == 1:
            ## Change low probability to estimations
            im = injectToProbLabels(im, 7, (wm_prob  < 50) * (im == 1))
            im = injectToProbLabels(im, 9, (csf_prob < 50) * (im == 2))
            im = injectToProbLabels(im, 8, (gm_prob  < 50) * (im == 3))
        else:
            ## Change low probability to estimations
            im = injectToProbLabels(im, 1, (wm_prob  > 50) * (im == 1))
            im = injectToProbLabels(im, 2, (csf_prob > 50) * (im == 2))
            im = injectToProbLabels(im, 3, (gm_prob  > 50) * (im == 3))
"""
    final_lbl_map = sitk.Cast(im * sitk.Cast(crbl_map, sitk.sitkInt8), sitk.sitkInt8)
    # final_lbl_map[29,160,79] = 6
    out_file = os.path.join(base_output_dir, output_prefix + "_auto_seg.nrrd")
    print("Saving file {0}".format(out_file))
    sitk.WriteImage(final_lbl_map, out_file)



if __name__ == "__main__":
    import sys
    import time

    start = time.time()

    # command line usage
    # "/Shared/johnsonhj/HDNI/20170425_CerebellumSegmentation/EditorHelperML" "/Shared/johnsonhj/HDNI/20170425_CerebellumSegmentation/EditorHelperML/data/t1_average_BRAINSABC.nii.gz" "/Shared/johnsonhj/HDNI/20170425_CerebellumSegmentation/EditorHelperML/data/t2_average_BRAINSABC.nii.gz" "/Shared/johnsonhj/HDNI/20170425_CerebellumSegmentation/EditorHelperML/mgi.nii.gz" "/Shared/johnsonhj/HDNI/20170425_CerebellumSegmentation/EditorHelperML/18_manual_seg.nrrd" "/Shared/johnsonhj/HDNI/20170425_CerebellumSegmentation/EditorHelperML/JointFusion_HDAtlas20_2015_dustCleaned_label.nii.gz" "test_case"
    base_output_dir = sys.argv[1]

    t1_intensity_image = sys.argv[2]
    t2_intensity_image = sys.argv[3]
    feature_image = sys.argv[4]


    crbl_samples_image = sys.argv[5]
    crbl_mask_image = sys.argv[6]
    output_prefix = sys.argv[7]

    updateMask(t1_intensity_image, t2_intensity_image, feature_image,
               crbl_samples_image, crbl_mask_image,
               base_output_dir, output_prefix)
    # print("File {0} written.".format(out_file))
    print("done")
    end = time.time()
    print("Elapsed Time: {0}".format(end - start))

