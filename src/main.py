from __future__ import division, print_function, absolute_import
import sys
import config
import datax
from datax.get_data import get_data
from datax.cerebellum_mri_data_bunch import CerebellumMriDataBunch
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
    
    data_obj = CerebellumMriDataBunch(base_dir, [1, 2, 3, 4, 5])
    x, y = data_obj.get_data_by_ids([1, 2, 3, 4, 5])

    if do_train:
        model.train(x, y, None, None)
    
    if do_predict:
        model.predict(data.all_imgs[0], testX)
