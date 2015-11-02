#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

fully = False
patch_size = 51

def generate_nnet(feats):
    """Generate a neural network.

    Parameters
    ----------
    feats : list with at least one feature vector

    Returns
    -------
    Neural network object
    """
    # Load it here to prevent crash of --help when it's not present
    import lasagne
    from lasagne import layers
    from lasagne.updates import nesterov_momentum
    from nolearn.lasagne import NeuralNet

    input_shape = (None,
                   feats[0].shape[0],
                   feats[0].shape[1],
                   feats[0].shape[2])
    logging.info("input shape: %s", input_shape)
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.Conv2DLayer),
                ('hidden2', layers.Conv2DLayer),
                ('pool', layers.MaxPool2DLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=input_shape,
        hidden_num_filters=10,
        hidden_filter_size=(5, 5),
        hidden_pad='same',
        hidden2_num_filters=10,
        hidden2_filter_size=(5, 5),
        hidden2_pad='same',
        pool_pool_size=(2, 2),
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=2,

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        max_epochs=10,
        verbose=1,)
    return net1
