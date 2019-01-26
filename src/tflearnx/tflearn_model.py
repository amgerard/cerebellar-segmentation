from __future__ import division, print_function, absolute_import
import sys
import tflearn
import tensorflow as tf

import tflearn.data_flow as tfdf
from tflearnx.data_flow_alex import FeedDictFlowAlex
tfdf.FeedDictFlow = FeedDictFlowAlex

import tflearnx.cnn_3d as cnn_3d
import tflearnx.cnn_2d as cnn_2d
#import evaluate
#from predict_mask import save_mask_old, save_mask

import numpy as np

class TflearnModel(object):

    def __init__(self, name, is_3d):
        
        self.name = name

        # create network and model
        device = '/gpu:0'
        with tf.device(device):
            network = cnn_3d.get_network() if is_3d else cnn_2d.get_network()

        self.model = tflearn.DNN(
            network,
            tensorboard_verbose=2)#,
            #tensorboard_dir=logs_dir,
            #checkpoint_path=cp_dir)

    
    def load_from_file(self, model_name):
        # load model
        if model_name != '':
            self.model.load(model_name)

    
    def train(self, X, Y, testX, testY):
        # train
        print('start training')
        print('X:', X.shape)
        print('Y:', Y.shape)
        trn_x = {'input': X}
        trn_y = {'target': Y}
        val_set = ({'input': testX}, {'target': testY})
        do_val = False
        try:
            if do_val:
                self.model.fit(
                    trn_x,
                    trn_y,
                    validation_set=val_set,
                    n_epoch=n_epochs,
                    snapshot_step=100,
                    show_metric=True,
                    run_id=rid)
            else:
                self.model.fit(
                    trn_x,
                    trn_y,
                    n_epoch=2,
                    snapshot_step=500,
                    show_metric=True,
                    run_id='1234')
        except KeyboardInterrupt:
            print('training interrupted!!!')

        # save model
        print('save model')
        self.model.save('models/' + self.name + ".tfl")


    def predict(self):
        # predict
        if do_predict:
            import numpy as np
            print('predict and save mask')
            y_pred = evaluate.predict(model, testX)
            #y_pred = np.loadtxt('/home/amgerard/src/CerebellumChallenge/cnn_src/b1_y_pred_iter5.txt')
            save_mask(data.all_imgs[0], testX, y_pred)
            #np.savetxt('/home/amgerard/src/CerebellumChallenge/cnn_src/b5_y_pred_iter6.txt', y_pred)


    def crap(self):
        if do_explore:
            with model.session.as_default():
                for v in tf.trainable_variables():
                    np_arr = v.eval()
                    print(v.name, np_arr.shape)
                    np.save('/home/amgerard/src/CerebellumChallenge/cnn_src/model_np/' + v.name.replace('/','_') + ".npy", np_arr)
            #import explore_model
            #explore_model.explore(data.all_imgs[2], model)

        # eval
        if do_eval:
            print('evaluate')
            print(testX.shape, testY.shape)
            #y_pred = evaluate.predict(model, testX)
            evaluate.evaluate_model(model, testX, testY)
            #evaluate.evaluate_model(model, X, Y)
