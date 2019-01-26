import SimpleITK as sitk


def resample_alex(output_path, nearest_neighbor=False, im_path='brain1_t1_cropped.nrrd'):
    im = sitk.ReadImage(im_path)
    origSpacing = im.GetSpacing();
    outvoxelsize = 1
    newSpacing = [0,0,0]
    newSpacing[0] = outvoxelsize;
    newSpacing[1] = outvoxelsize;
    newSpacing[2] = outvoxelsize;

    origSize = im.GetSize()#GetLargestPossibleRegion().GetSize();
    newSize = [0,0,0]
    newSize[0] = origSize[0]*origSpacing[0]/newSpacing[0];
    newSize[1] = origSize[1]*origSpacing[1]/newSpacing[1];
    newSize[2] = origSize[2]*origSpacing[2]/newSpacing[2];
    newSize = [int(x) for x in newSize]

    print("Resampling image")
    resample = sitk.ResampleImageFilter()
    #resample.SetInput(im);
    #interp = sitk.sitkBSpline if not nearest_neighbor else 
    if nearest_neighbor:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    #transform = sitk.Transform(3, sitk.sitkIdentity) if not nearest_neighbor else 
    resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resample.SetSize(newSize)
    resample.SetOutputSpacing(newSpacing)
    resample.SetOutputOrigin(im.GetOrigin())
    resample.SetOutputDirection(im.GetDirection())

    print("Writing output... ")
    sitk.WriteImage(resample.Execute(im), output_path)


if __name__ == "__main__":
    out_dir = '/home/amgerard/src/CerebellumChallenge/CerebellumData/CrblData/1mm/'
    base_dir = '/home/amgerard/src/CerebellumChallenge/CerebellumData/CrblData/CRBLM_DeepLearning/'
    #resample_alex('b1_1mm_labels.nrrd', True, base_dir + 'brain1/labels/pred/b1_auto_seg_3d_100_iter10_init.nrrd')

    for b in ['brain1', 'brain2', 'brain3', 'brain4', 'brain5']:
        t1_path = base_dir + b + "/" + b + "_t1_cropped.nrrd"
        t2_path = base_dir + b + "/" + b + "_t2_cropped.nrrd"
        resample_alex(out_dir + b + "_t1_1mm.nrrd", False, t1_path)
        resample_alex(out_dir + b + "_t2_1mm.nrrd", False, t2_path)
        resample_alex(out_dir + b + "_lbls_1mm.nrrd", True, base_dir + b + "/labels/pred/" + b.replace("rain", "") + '_auto_seg_3d_100_iter10_init.nrrd')