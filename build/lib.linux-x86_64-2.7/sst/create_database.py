#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a neural network to classify (patch size)x(patch size) (3 color) patches
with street / no street in the center pixel.

If loss doesn't change after the first iterations, you have to re-run the
training.
"""

from __future__ import print_function

import inspect
import imp
from pkg_resources import resource_filename
import pickle
import sys
import os
import logging

import scipy
import numpy as np
import random

from . import utils


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def main(network_path, image_batch_size=100, stride=25,
         images_folder='roadC621/'):
    """
    Train a neural network with patches of patch_size x patch_size
    (as given via the module network_path).

    Parameters
    ----------
    network_path : str
        Path to a Python script with a function generate_nnet(feats) which
        returns a neural network
    image_batch_size : int
    stride : int
    """
    assert image_batch_size >= 1
    assert stride >= 1
    features, labels = load_data_raw_images(train_images_folder=images_folder)
    mem_size = (sys.getsizeof(42) * len(features) * features[0].size +
                sys.getsizeof(42) * len(labels) * labels[0].size)
    logging.info("Loaded %i data images with their labels (approx %s)",
                 len(features),
                 utils.sizeof_fmt(mem_size))
    nn_params = {'training': {'image_batch_size': image_batch_size,
                              'stride': stride}}

    logging.info("## Network: %s", network_path)
    network = imp.load_source('sst.network', network_path)
    logging.info("Fully network: %s", str(network.fully))
    nn_params['code'] = inspect.getsource(network)
    nn_params['fully'] = network.fully
    nn_params['patch_size'] = network.patch_size
    assert nn_params['patch_size'] > 0

    labeled_patches = get_patches(features[:1],
                                  labels[:1],
                                  nn_params=nn_params)

    feats, _ = get_features(labeled_patches, fully=nn_params['fully'])
    net1 = network.generate_nnet(feats)
    for block in range(0, len(features), image_batch_size):
        from_img = block
        to_img = block + image_batch_size
        logging.info("Training on batch %i - %i of %i total",
                     from_img,
                     to_img,
                     len(features))
        labeled_patches = get_patches(features[from_img:to_img],
                                      labels[from_img:to_img],
                                      nn_params=nn_params,
                                      stride=stride)
        logging.info(("labeled_patches[0].shape: %s , "
                      "labeled_patches[1].shape: %s"),
                     labeled_patches[0].shape,
                     labeled_patches[1].shape)
        net1 = train_nnet(labeled_patches, net1, fully=nn_params['fully'])

    model_pickle_name = 'nnet1-trained.pickle'
    utils.serialize_model(net1,
                          filename=model_pickle_name,
                          parameters=nn_params)


def load_data_raw_images(serialization_path='data.pickle',
                         train_images_folder='data_road/roadC621/'):
    """Load color images (3 channels) and labels (as images).

    Returns
    -------
    tuple : (featurevector list, label list)
    """
    logging.info("Start loading data...")
    data_source = serialization_path + ".npz"

    if not os.path.exists(data_source):
        # build lists of files which will be read
        path_data = os.path.join(os.environ['DATA_PATH'],
                                 train_images_folder,
                                 "image_2/")
        files_data = [os.path.join(path_data, f)
                      for f in sorted(os.listdir(path_data))
                      if f.endswith('.png')]

        path_gt = os.path.join(os.environ['DATA_PATH'],
                               train_images_folder,
                               "gt_image_2/")
        files_gt = [os.path.join(path_gt, f)
                    for f in sorted(os.listdir(path_gt))
                    if f.endswith('.png')]
        if not os.path.isfile('training.pickle') or \
           not os.path.isfile('testing.pickle'):
            logging.info("Write training.pickle and testing.pickle")
            write_files(files_data, files_gt)

        filelist_tuples = read_filelist('training.pickle')
        files_data, files_gt = [], []
        for file_data, file_gt in filelist_tuples:
            files_data.append(file_data)
            files_gt.append(file_gt)

        # read files (data first)
        print("Start reading images: ", end='')
        colored_image_features = []
        for img_path in files_data:
            print('.', end='')
            ac = utils.load_color_image_features(img_path)
            if(ac.shape[0] == 188):  # TODO: Why is this skipped?
                colored_image_features.append(ac)
        print('')
        xs_colored = np.array(colored_image_features, copy=False)

        # read grayscale groundtruth
        yl = []
        for f in files_gt:
            img = scipy.misc.imread(f)
            if(img.shape[0] != 188):  # TODO: Why is this skipped?
                continue
            new_img = np.zeros(img.shape)
            for i, row in enumerate(img):
                for j, pixel in enumerate(row):
                    new_img[i][j] = (105 == pixel)
            yl.append(new_img)
        yl = np.array(yl)

        assert len(xs_colored) == len(yl), "len(xs_colored) != len(yl)"
        for i, (X, y) in enumerate(zip(xs_colored, yl), start=1):
            logging.info("Get labels (%i/%i)...", i, len(yl))
            assert X.shape[:2] == y.shape, \
                ("X.shape[1:]=%s and y.shape=%s" %
                 (X.shape[:2], y.shape))
            assert min(y.flatten()) == 0.0, \
                ("min(y)=%s" % str(min(y.flatten())))
            assert max(y.flatten()) == 1.0, \
                ("max(y)=%s" % str(max(y.flatten())))
        np.savez(serialization_path, xs_colored, yl)
    else:
        logging.info("!! Loaded pickled data" + "!" * 80)
        logging.info("Data source: %s", data_source)
        logging.info("This implies same test / training split as before.")
        npzfile = np.load(data_source)
        xs_colored = npzfile['arr_0']
        yl = npzfile['arr_1']
    return (xs_colored, yl)


def write_files(files_data, files_gt, training_ratio=0.8):
    """Split list of images into training and test set.

    Parameters
    ----------
    files_data : list of str
        Paths to data files
    files_gt : list of str
        Paths to ground truth (same order as data images)
    """
    zipped = list(zip(files_data, files_gt))
    random.shuffle(zipped)
    assert 0.1 <= training_ratio <= 1.0, 'wrong training ratio'
    split_to = int(len(zipped) * training_ratio)
    with open('training.pickle', 'wb') as f:
        pickle.dump(zipped[:split_to], f)
    with open('testing.pickle', 'wb') as f:
        pickle.dump(zipped[split_to:], f)


def read_filelist(filename):
    """
    Parameters
    ----------
    filename : str
        Path to a .pickle file

    Returns
    -------
    list of tuples [(path to data, path to ground truth)]
    """
    with open(filename, 'rb') as f:
        files = pickle.load(f)
    return files


def get_patches(xs, ys, nn_params, stride=49):
    """Get a list of tuples (patch, label), where label is int
    (1=street, 0=no street) and patch is a 2D-array of floats.

    Parameters
    ----------
    xs : list
        Each element is an image with 3 channels (RGB), but normalized to
        [-1, 1]
    ys : list
        Each element is either 0 or 1
    nn_params : dict
        All relevant parameters of the model (e.g. patch_size and fully)
    stride : int
        The smaller this value, the more patches will be created.

    Returns
    -------
    tuple : (patches, labels)
        Two lists of same length. Patches is
    """
    patch_size = nn_params['patch_size']
    fully = nn_params['fully']
    assert stride >= 1, "Stride must be at least 1"
    assert (patch_size) >= 1, "Patch size has to be >= 1"
    assert patch_size % 2 == 1, "Patch size should be odd"
    assert xs[0].shape[0] >= patch_size and xs[0].shape[1] >= patch_size, \
        ("Patch is too big for this image: img.shape = %s" % str(xs[0].shape))
    logging.info("Get patches of size: %i", patch_size)
    patches, labels = [], []
    for X, y in zip(xs, ys):
        px_left_patchcenter = (patch_size - 1) / 2
        start_x = px_left_patchcenter
        end_x = X.shape[0] - px_left_patchcenter
        start_y = start_x
        end_y = X.shape[1] - px_left_patchcenter
        for patch_center_x in range(start_x, end_x + 1, stride):
            for patch_center_y in range(start_y, end_y + 1, stride):
                if fully:
                    # Get Labels of the patch and flatt it to 1D
                    # x1 = patch_center_x - px_left_patchcenter
                    # x2 = patch_center_x + px_left_patchcenter + 1
                    # y1 = patch_center_y - px_left_patchcenter
                    # y2 = patch_center_y + px_left_patchcenter + 1
                    l = y[patch_center_x - px_left_patchcenter:
                          patch_center_x + px_left_patchcenter + 1,
                          patch_center_y - px_left_patchcenter:
                          patch_center_y + px_left_patchcenter + 1]

                    labels.append(l.flatten())

                    # Get patch from original image
                    patches.append(X[patch_center_x - px_left_patchcenter:
                                     patch_center_x + px_left_patchcenter + 1,
                                     patch_center_y - px_left_patchcenter:
                                     patch_center_y + px_left_patchcenter + 1,
                                     :])
                else:
                    labels.append(y[patch_center_x][patch_center_y])
                    # Get patch from original image
                    patches.append(X[patch_center_x - px_left_patchcenter:
                                     patch_center_x + px_left_patchcenter + 1,
                                     patch_center_y - px_left_patchcenter:
                                     patch_center_y + px_left_patchcenter + 1,
                                     :])
    assert len(patches) == len(labels), "len(patches) != len(labels)"
    logging.info("%i patches were generated.", len(patches))
    if fully:
        return (np.array(patches, dtype=np.float32),
                np.array(labels, dtype=np.float32))
    else:
        return (np.array(patches, dtype=np.float32),
                np.array(labels, dtype=np.int32))


def get_features(labeled_patches, fully=False):
    """Get ready-to-use features from labeled patches

    Parameters
    ----------
    labeled_patches : tuple (patches, labels)

    Returns
    -------
    tuple (feats, y)
        list of feature vectors and list of labels
    """
    feats = labeled_patches[0]
    y = labeled_patches[1]

    if not fully:
        logging.info("Street feature vectors: %i",
                     sum([1 for label in y if label == 0]))
        logging.info("Non-Street feature vectors: %i",
                     sum([1 for label in y if label == 1]))
    logging.info("Feature vectors: %i", len(y))

    # original shape: (25, 25, 3)
    # desired shape: (3, 25, 25)
    feats_new = []

    for ac in feats:
        c = []
        c.append(ac[:, :, 0])
        c.append(ac[:, :, 1])
        c.append(ac[:, :, 2])
        feats_new.append(c)
    feats = np.array(feats_new, dtype=np.float32)
    return (feats, y)


def train_nnet(labeled_patches, net1, fully=False):
    """Train a neural network classifier on the patches.

    Parameters
    ----------
    labeled_patches : tuple (patches, labels)

    Returns
    -------
    trained classifier
    """
    feats, y = get_features(labeled_patches, fully)
    net1.fit(feats, y)
    return net1


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stride",
                        dest="stride",
                        default=20,
                        type=int,
                        help=("stride to run over the images - influences "
                              "amount of training data"))
    parser.add_argument("--batchsize",
                        dest="batchsize",
                        default=100,
                        type=int,
                        help=("batches"))
    parser.add_argument("--network",
                        dest="network",
                        default=resource_filename('sst.networks',
                                                  'fully_simple.py'),
                        type=str,
                        help=("path to a Python file with "
                              "generate_nnet(feats)"))
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(network_path=args.network,
         stride=args.stride,
         image_batch_size=args.batchsize)