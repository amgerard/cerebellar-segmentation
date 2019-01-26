from __future__ import division, print_function, absolute_import
import sys
#import evaluate
import datax
from datax.get_data import get_data
from tflearnx.tflearn_model import TflearnModel
from predict_mask import save_mask_old, save_mask

import numpy as np

if __name__ == '__main__':

    base_dir = sys.argv[1]

    # knobs
    n_epochs = 3
    name = '3d_iter10_sans_b5'#'3d_iter10' #'2d_2im_6' #
    do_train = True
    do_val = False
    do_eval = True
    do_predict = False
    do_explore = False
    datax.do_3d = True

    model_name = 'models/' + name + '.tfl'

    # saving?
    logs_dir = 'logs/' + name + '/'
    cp_dir = 'chpts/' + name + '/'
    rid = name

    # get data
    if not base_dir:
        base_dir = '/home/amgerard/src/CerebellumChallenge/CerebellumData/CrblData/CRBLM_DeepLearning/'
    X, Y, testX, testY = get_data(base_dir)

    # create network and model
    model = TflearnModel(name, datax.do_3d)

    # load model
    #if model_name != '':
    #    model.load_from_file(model_name)

    # train
    if do_train:
        model.train(X, Y, testX, testY)

    # predict
    if do_predict:
        model.predict(data.all_imgs[0], testX)

    if do_explore:
        pass

    # eval
    if do_eval:
        pass
