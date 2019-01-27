from __future__ import division, print_function, absolute_import
import sys
import tflearn
import tensorflow as tf

import tflearn.data_flow as tfdf
from tflearnx.data_flow_alex import FeedDictFlowAlex
tfdf.FeedDictFlow = FeedDictFlowAlex

import tflearnx.cnn_3d as cnn_3d
import tflearnx.cnn_2d as cnn_2d
import evaluate
from predict_mask import save_mask

import numpy as np

class TflearnModel(object):

    def __init__(self, name, patch_num_dim, n_epochs):
        
        self.name = name
        logs_dir = 'logs/' + name + '/'
        cp_dir = 'chpts/' + name + '/'
        self.n_epochs = n_epochs

        device = '/gpu:0'
        with tf.device(device):
            network = cnn_3d.get_network() if patch_num_dim == 3 else cnn_2d.get_network()

        self.model = tflearn.DNN(
            network,
            tensorboard_verbose=2,
            tensorboard_dir=logs_dir,
            checkpoint_path=cp_dir)

    
    def load(self):
        model_path = 'models/' + self.name + '.tfl'
        self.model.load(model_path)

    
    def train(self, X, Y, testX, testY):
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
                    n_epoch=self.n_epochs,
                    snapshot_step=100,
                    show_metric=True,
                    run_id=self.name)
            else:
                self.model.fit(
                    trn_x,
                    trn_y,
                    n_epoch=self.n_epochs,
                    snapshot_step=500,
                    show_metric=True,
                    run_id=self.name)
        except KeyboardInterrupt:
            print('Training interrupted.')
        
        print('save model')
        self.model.save('models/' + self.name + ".tfl")


    def predict(self, image, x_test):
        y_pred = evaluate.predict(self.model, x_test)
        save_mask(image, x_test, y_pred)


    def explore(self):
        with self.model.session.as_default():
            for v in tf.trainable_variables():
                np_arr = v.eval()
                print(v.name, np_arr.shape)
                np.save('/home/amgerard/src/CerebellumChallenge/cnn_src/model_np/' + v.name.replace('/','_') + ".npy", np_arr)
        #import explore_model
        #explore_model.explore(data.all_imgs[2], model)
