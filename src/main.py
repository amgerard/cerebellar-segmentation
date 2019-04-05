from __future__ import division, print_function, absolute_import
import sys
import config
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
    pred_out_file = "brain1_pred.nrrd"
    pred_brain_id = 1

    model = TflearnModel(name, config.patch_num_dim, n_epochs)

    if do_load_model:
        model.load()
    
    brain_ids = [1, 2, 3, 4, 5] if do_train else [pred_brain_id]
    data_bunch = CerebellumMriDataBunch(base_dir, brain_ids)
    config.data_bunch = data_bunch

    if do_train:
        x, y = data_bunch.get_data_by_ids(brain_ids)
        model.train(x, y, None, None)
    
    if do_predict:
        im = data_bunch.get_image(pred_brain_id)
        model.predict(im.get_image_arr(), im.get_testing_data(), im.get_image_t1(), pred_out_file)
