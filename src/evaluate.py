import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transform_patch import rotate_combine_normalize, extract_patch
import data
import config

p_sz = config.patch_size

def predict(model, testX):
    '''
    predict feat vects to test using model
    :param model: tensorflow model
    :param testX: npy feat vects (idx, rot, flip)
    :return: prediction for each patch
    '''
    pred = np.zeros([testX.shape[0]])
    subset_size = 200 if data.do_3d else 1000
    patches_subset = []
    good_idxs = []
    print(testX.shape, testX[:,0].min(), testX[:,0].max())
    for ix in range(testX.shape[0]):
        idxs = testX[ix, 0:3] # testX[ix, 1:4]
        im_idx = 2 # testX[ix, 0]

        i, j, k = idxs
        #patch = rotate_combine_normalize(idxs, [0, 0, 0], False)
        patch = extract_patch(im_idx, idxs)
        if patch is None:
            print('Null patch: ', idxs)
            continue
        if (data.do_3d and patch.shape == (p_sz,p_sz,p_sz,2)) or patch.shape == (p_sz,p_sz,2):
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


def evaluate_model(model, testX, testY):
    print('accuracy', model.evaluate(testX, testY))
    return

    ty = np.argmax(testY, axis=1)
    # print model.evaluate(testX,testY)
    #tp = np.argmax(model.predict(testX),axis=1)
    tp = predict(model, testX)
    conf_arr = confusion_matrix(ty, tp)
    print(conf_arr)

    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = ['HT', 'N', 'E', 'NET', 'ET']
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('confusion_matrix.png', format='png')
