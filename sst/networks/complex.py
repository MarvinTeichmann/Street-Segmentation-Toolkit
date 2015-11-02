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
    from lasagne import layers
    from nolearn.lasagne import NeuralNet
    from .shape import ReshapeLayer
    from .unpool import Unpool2DLayer

    input_shape = (None,
                   feats[0].shape[0],
                   feats[0].shape[1],
                   feats[0].shape[2])
    logging.info("input shape: %s", input_shape)

    # conv_filters = 32
    patch_size = 51
    deconv_filters = 32
    filter_sizes = 3

    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.Conv2DLayer),

                ('pool', layers.MaxPool2DLayer),
                ('flatten', ReshapeLayer),  # output_dense
                ('encode_layer', layers.DenseLayer),
                ('hidden2', layers.DenseLayer),  # output_dense
                ('unflatten', ReshapeLayer),
                ('hidden3', layers.Conv2DLayer),
                # ('unpool', Unpool2DLayer),
                # ('conv1', layers.Conv2DLayer),

                ('deconv', layers.Conv2DLayer),
                ('output_layer', ReshapeLayer),  # flatten output
                ],
        # layer parameters:
        input_shape=input_shape,
        hidden_num_filters=10,
        hidden_filter_size=(3, 3),
        hidden_border_mode='same',

        pool_pool_size=(3, 3),
        flatten_shape=(([0], -1)),  # not sure if necessary?
        encode_layer_num_units = 500,
        hidden2_num_units= deconv_filters * (patch_size + filter_sizes - 1) ** 2 / 4,
        unflatten_shape=(([0], 8, 53, 53)),
        hidden3_num_filters=1,
        hidden3_filter_size=(3, 3),
        hidden3_border_mode='valid',
        # unpool_ds=(3, 3),
        # conv1_num_filters=10,
        # conv1_filter_size=(3, 3),
        # conv1_border_mode='same',

        deconv_num_filters=1,
        deconv_filter_size=(feats[0].shape[1], feats[0].shape[2]),
        deconv_border_mode='same',
        output_layer_shape = (([0], -1)),
        regression=True,

        # optimization method:
        # update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        max_epochs=20,
        verbose=1,)
    return net1
