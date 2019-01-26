from __future__ import division, print_function, absolute_import
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression


def get_network(do_dropout=True):
    # input images
    network = input_data(shape=[None, 25, 25, 25, 2], name='input')

    network = conv_3d(network, 50, 3, activation='relu', regularizer="L2")
    network = max_pool_3d(network, 2)
    network = batch_normalization(network)  # local_response_normalization

    network = conv_3d(network, 75, 3, activation='relu', regularizer="L2")
    #network = max_pool_3d(network, 2)
    network = batch_normalization(network)

    network = conv_3d(network, 100, 3, activation='relu', regularizer="L2")
    network = max_pool_3d(network, 2)
    network = batch_normalization(network)

    # fully connected layers
    network = fully_connected(network, 64, activation='relu')
    if do_dropout:
        network = dropout(network, 0.5)

    #network = local_response_normalization(network)
    network = batch_normalization(network)
    network = fully_connected(network, 64, activation='relu')
    if do_dropout:
        network = dropout(network, 0.5)
    network = batch_normalization(network)

    # softmax + output layers
    network = fully_connected(network, 3, activation='softmax', name='soft')
    network = regression(
        network,
        optimizer='adam',
        learning_rate=0.00001,
        loss='categorical_crossentropy',
        name='target',
        batch_size=75)  # 0.000005
    return network
