from __future__ import division, print_function, absolute_import
from datax.cerebellum_mri_data import CerebellumMriData
import numpy as np

class CerebellumMriDataBunch(object):

    def __init__(self, base_dir, brain_ids):

        self.base_dir = base_dir
        self.brain_ids = brain_ids
        self.images = {id:self.__create_brain_data_object(id) for id in self.brain_ids}


    def __create_brain_data_object(self, brain_id):
        brain_path = self.base_dir + 'brain{0}/brain{0}_'
        t1_path = brain_path + 't1_cropped.nrrd'
        t2_path = brain_path + 't2_cropped.nrrd'
        ground_truth_path = self.base_dir + 'brain{0}/labels/brain{0}_' + 'labels.nrrd'
        cerebellum_mask_path = brain_path + 'mask_cropped.nrrd'

        t1, t2 = t1_path.format(brain_id), t2_path.format(brain_id)
        gt, crbl = ground_truth_path.format(brain_id), cerebellum_mask_path.format(brain_id)
        return CerebellumMriData(t1, t2, crbl, gt)

    
    def get_data_by_ids(self, brain_ids):
        x, y = np.zeros((1,8)), np.zeros((1,3))

        for i in brain_ids:
            _x, _y = self.images[i].get_training_data(i)
            x = np.concatenate([x, _x], axis=0)
            y = np.concatenate([y, _y], axis=0)

        randTrn = np.random.randint(0, 3, x.shape[0])
        x = x[randTrn == 1]
        y = y[randTrn == 1]

        #x, y = get_shuffled(x, y)

        print('done loading train: ', x[:5,:], y[:5])
        print('done loading train: ', x.shape, y.shape)

        return x, y