# -*- coding: utf-8 -*-

import logging

fully = True
patch_size = 51


def generate_nnet(feats):
    """Generate a neural network.
    (TODO: Difference between "fully" and non-fully)

    Parameters
    ----------
    feats : list with at least one feature vector

    Returns
    -------
    Neural network object
    """
    # Load it here to prevent crash of --help when it's not present
    from lasagne import layers
    from nolearn.lasagne import NeuralNet
    from .shape import ReshapeLayer

    input_shape = (None,
                   feats[0].shape[0],
                   feats[0].shape[1],
                   feats[0].shape[2])
    logging.info("input shape: %s", input_shape)
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.Conv2DLayer),
                ('hidden2', layers.Conv2DLayer),
                ('flatten', ReshapeLayer),  # flatten output
                ],
        # layer parameters:
        input_shape=input_shape,
        hidden_num_filters=10,
        hidden_filter_size=(3, 3),
        hidden_pad='same',
        hidden2_num_filters=1,
        hidden2_filter_size=(feats[0].shape[1], feats[0].shape[2]),
        hidden2_pad='same',
        flatten_shape = (([0], -1)),
        regression=True,

        # optimization method:
        update_learning_rate=0.01,
        update_momentum=0.9,

        max_epochs=10,
        verbose=1,)
    return net1
