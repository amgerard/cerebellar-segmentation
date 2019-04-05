import random
import numpy as np

def get_shuffled(x_arr, y_arr):
    x_and_y = [np.append(x_arr[i],y_arr[i]) for i in range(x_arr.shape[0])]
    np.random.shuffle(x_and_y)
    x_dim1 = x_arr.shape[1]
    x_shuffle = [a[:x_dim1] for a in x_and_y]
    y_shuffle = [a[x_dim1:] for a in x_and_y]
    x_shuffle = np.array(x_shuffle)
    y_shuffle = np.array(y_shuffle)
    return x_shuffle, y_shuffle

if __name__ == '__main__':
    pass
