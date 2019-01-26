from __future__ import division, print_function, absolute_import
from datax.data import get_trn_data, get_test_data, get_shuffled
import numpy as np

def get_data(base_dir, validation_brain_id=-1):

    brain_path = base_dir + 'brain{0}/brain{0}_'
    t1_path = brain_path + 't1_cropped.nrrd'
    t2_path = brain_path + 't2_cropped.nrrd'
    ground_truth_path = base_dir + 'brain{0}/labels/brain{0}_' + 'labels.nrrd'
    cerebellum_mask_path = brain_path + 'mask_cropped.nrrd'

    train_brain_ids = [i for i in range(1, 6) if i != validation_brain_id and i < 4]

    X, Y = np.zeros((1,8)), np.zeros((1,3))
    testX, testY = np.array([]), np.array([])

    for i in train_brain_ids:
        t1, t2 = t1_path.format(i), t2_path.format(i)
        gt, crbl = ground_truth_path.format(i), cerebellum_mask_path.format(i)
        print(t1, t2, gt, crbl)
        x, _, y, _ = get_trn_data(t1, t2, gt, crbl, False)
        print('brain{0} shape: '.format(i), x.shape, y.shape)
        X = np.concatenate([X, x], axis=0)
        Y = np.concatenate([Y, y], axis=0)

    randTrn = np.random.randint(0, 3, X.shape[0])
    X = X[randTrn == 1]
    Y = Y[randTrn == 1]
    #randVal = np.random.randint(0, 4, testX.shape[0])
    #testX = testX[randVal == 1]
    #testY = testY[randVal == 1]

    X,Y = get_shuffled(X,Y)

    print('done loading train: ', X[:5,:], Y[:5])
    print('done loading train: ', X.shape, Y.shape)
    print('done loading test: ', testX.shape, testY.shape)

    #if do_predict or do_explore:
    #    testX = get_test_data(t1_path_b5, t2_path_b5, cer_path_b5)
    #    print('test data shape: ', testX.shape)

    return X, Y, testX, testY
