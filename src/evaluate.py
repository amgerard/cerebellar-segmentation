import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datax.transform_patch import rotate_combine_normalize, extract_patch
import datax.data as data
import config

p_sz = config.patch_size

def predict(model, image_arr, testX):
    '''
    predict feat vects to test using model
    :param model: tensorflow model
    :param testX: npy feat vects (idx, rot, flip)
    :return: prediction for each patch
    '''
    pred = np.zeros([testX.shape[0]])
    subset_size = 200 if config.patch_num_dim == 3 else 1000
    patches_subset = []
    good_idxs = []
    print(testX.shape, testX[:,0].min(), testX[:,0].max())
    for ix in range(testX.shape[0]):
        idxs = testX[ix, 0:3]
        patch = extract_patch(image_arr, idxs)
        if patch is None:
            print('Null patch: ', idxs)
            continue
        if (config.patch_num_dim == 3 and patch.shape == (p_sz,p_sz,p_sz,2)) or patch.shape == (p_sz,p_sz,2):
            patches_subset.append(patch)
            good_idxs.append(ix)
        else:
            print('Bad patch shape: ', patch.shape)

        if len(patches_subset) == subset_size or ix == testX.shape[0] - 1:
            probs = model.predict(patches_subset)
            pp = np.argmax(probs, axis=1)
            pred[good_idxs] = pp + 1

            patches_subset = []
            good_idxs = []
        if ix % 100000 == 0:
            print('Predict progress: ', ix)

    return pred

