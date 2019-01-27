from __future__ import division, print_function, absolute_import
import sys
import config
import datax
from datax.get_data import get_data
from tflearnx.tflearn_model import TflearnModel

if __name__ == '__main__':

    base_dir = '/home/amgerard/src/CerebellumChallenge/CerebellumData/CrblData/CRBLM_DeepLearning/'
    if len(sys.argv) > 0:
        base_dir = sys.argv[1]

    n_epochs = 2
    name = '3d_iter10_sans_b5'
    do_load_model = False
    do_train = True
    do_predict = False

    model = TflearnModel(name, config.patch_num_dim, n_epochs)

    if do_load_model:
        model.load()

    X, Y, testX, testY = get_data(base_dir)

    if do_train:
        model.train(X, Y, testX, testY)
    
    if do_predict:
        model.predict(data.all_imgs[0], testX)
