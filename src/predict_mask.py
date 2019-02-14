import os
import multiprocessing as mp
import SimpleITK as sitk
import numpy as np

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
