import tflearn
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import config

sb,ad = config.patch_size_adder,config.patch_size_subtracter
def explore(img_arr, model):
    i,j,k = (87, 100, 40)

    with model.session.as_default():
        for v in tf.trainable_variables():
            np_arr = v.eval()
            print(v.name, np_arr.shape)
            np.savetxt('/home/amgerard/src/CerebellumChallenge/cnn_src/model_np/' + v.name + ".txt", np_arr)
    return
    pat = img_arr[i-sb:i+ad,j-sb:j+ad,k,0]
    print('ALEX2',pat.shape)
    plt.figure()
    imgplot = plt.imshow(pat)
    plt.show()