#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Segment pixel-wise street/not street for a single image with a lasagne model.
"""
import logging
import sys
import time


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

import scipy
import numpy as np
import pickle

# sst modules
from . import utils


def main(image_path, output_path, model_path_trained, stride,
         hard_classification=True):
    with Timer() as t:
        nn, parameters = utils.deserialize_model(model_path_trained)
    assert stride <= parameters['patch_size']
    logging.info("Patch size: %i", parameters['patch_size'])
    logging.info("Fully: %s", str(parameters['fully']))
    logging.info("Stride: %i", stride)
    logging.info("=> elasped deserialize model: %s s", t.secs)
    with Timer() as t:
        result = eval_net(trained=nn,
                          photo_path=image_path,
                          parameters=parameters,
                          stride=stride,
                          hard_classification=hard_classification)
    logging.info("=> elasped evaluating model: %s s", t.secs)
    scipy.misc.imsave(output_path, result)
    utils.overlay_images(image_path, result, output_path,
                         hard_classification=hard_classification)


def eval_net(trained,
             photo_path,
             parameters=None,
             stride=10,
             hard_classification=True,
             verbose=False):
    """
    Parameters
    ----------
    trained : theano expression
        A trained neural network
    photo_path : string
        Path to the photo which will get classified
    parameters : dict
        Parameters relevant for the model such as patch_size
    stride : int
    hard_classification : bool
        If True, the image will only show either street or no street.
        If False, the image will show probabilities.
    verbose : bool
    """

    patch_size = parameters['patch_size']
    fully = parameters['fully']

    # read images
    feats = utils.load_color_image_features(photo_path)
    orig_dimensions = feats.shape

    patches = []
    px_left_patchcenter = (patch_size - 1) / 2

    height, width = feats.shape[0], feats.shape[1]
    if fully:
        to_pad_width = (patch_size - width) % stride
        to_pad_height = (patch_size - height) % stride

        # Order of to_pad_height / to_pad_width tested with scipy.misc.imsave
        feats = np.pad(feats,
                       [(to_pad_height, 0),
                        (to_pad_width / 2, to_pad_width - (to_pad_width / 2)),
                        (0, 0)],
                       mode='edge')
    else:
        feats = np.pad(feats,
                       [(px_left_patchcenter, px_left_patchcenter),
                        (px_left_patchcenter, px_left_patchcenter),
                        (0, 0)],
                       mode='edge')
    start_x = px_left_patchcenter
    end_x = feats.shape[0] - px_left_patchcenter
    start_y = start_x
    end_y = feats.shape[1] - px_left_patchcenter
    new_height, new_width = 0, 0
    for patch_center_x in range(start_x, end_x, stride):
        new_height += 1
        for patch_center_y in range(start_y, end_y, stride):
            if new_height == 1:
                new_width += 1
            # Get patch from original image
            new_patch = feats[patch_center_x - px_left_patchcenter:
                              patch_center_x + px_left_patchcenter + 1,
                              patch_center_y - px_left_patchcenter:
                              patch_center_y + px_left_patchcenter + 1,
                              :]
            patches.append(new_patch)

    if verbose:
        logging.info("stride: %s", stride)
        logging.info("patch_size: %i", patch_size)
        logging.info("fully: %s", str(fully))
        logging.info("Generated %i patches for evaluation", len(patches))
    to_classify = np.array(patches, dtype=np.float32)

    x_new = []

    for ac in to_classify:
        c = []
        c.append(ac[:, :, 0])
        c.append(ac[:, :, 1])
        c.append(ac[:, :, 2])
        x_new.append(c)
    to_classify = np.array(x_new, dtype=np.float32)

    if hard_classification:
        result = trained.predict(to_classify)
    else:
        result = trained.predict_proba(to_classify)
        if not fully:
            result_vec = np.zeros(result.shape[0])
            for i, el in enumerate(result):
                result_vec[i] = el[1]
            result = result_vec

    # Compute combined segmentation of image
    if fully:
        result = result.reshape(result.shape[0], patch_size, patch_size)
        result = result.reshape(new_height, new_width, patch_size, patch_size)

        # Merge patch classifications into a single image (result2)
        result2 = np.zeros((height, width))

        left_px = (patch_size - stride) / 2
        right_px = left_px + stride  # avoid rounding problems with even stride

        offset = {'h': 0, 'w': 0}

        if verbose:
            logging.info("new_height=%i, new_width=%i", new_height, new_width)
            logging.info("result.shape = %s", str(result.shape))
        for j in range(0, new_height):
            for i in range(0, new_width):
                if i == 0:
                    left_margin_px = to_pad_width / 2
                    right_margin_px = right_px
                elif i == new_width - 1:
                    left_margin_px = left_px
                    # TODO (TOTHINK): -1: it's a kind of magic magic...
                    # seems to do the right thing...
                    right_margin_px = patch_size - (to_pad_width -
                                                    (to_pad_width / 2)) - 1
                else:
                    left_margin_px = left_px
                    right_margin_px = right_px
                if j == 0:
                    top_px = to_pad_height
                    bottom_px = right_px
                elif j == new_height - 1:
                    top_px = left_px
                    bottom_px = patch_size
                else:
                    top_px = left_px
                    bottom_px = right_px

                # TOTHINK: no +1?
                to_write = result[j, i,
                                  top_px:(bottom_px),
                                  left_margin_px:(right_margin_px)]

                if i == 0 and j == 0:
                    offset['h'] = to_write.shape[0]
                    offset['w'] = to_write.shape[1]

                start_h = (offset['h'] + (j - 1) * stride) * (j != 0)
                start_w = (offset['w'] + (i - 1) * stride) * (i != 0)
                result2[start_h:start_h + to_write.shape[0],
                        start_w:start_w + to_write.shape[1]] = to_write

        if hard_classification:
            result2 = np.round((result2 - np.amin(result2)) /
                               (np.amax(result2) - np.amin(result2)))

        result2 = result2 * 255

        return result2
    else:
        result = result.reshape((new_height, new_width)) * 255

        # Scale image to correct size
        result = scale_output(result, orig_dimensions)
        return result


def eval_pickle(trained, parameters, test_pickle_path, stride=1):
    """
    Parameters
    ----------
    trained : theano expression
        A trained neural network
    parameters : dict
        parameters relevant for the model (e.g. patch size)
    test_pickle_path : str
        Path to a pickle file
    """
    with open(test_pickle_path, 'rb') as f:
        list_tuples = pickle.load(f)

    total_results = {'tp': 0,
                     'tn': 0,
                     'fp': 0,
                     'fn': 0}
    relative_results = {'tp': 0.0,
                        'tn': 0.0,
                        'fp': 0.0,
                        'fn': 0.0}
    for i, (data_image_path, gt_image_path) in enumerate(list_tuples):
        logging.info("Processing image: %s of %s", i + 1, len(list_tuples))
        result = eval_net(trained,
                          photo_path=data_image_path,
                          parameters=parameters,
                          stride=stride)
        tmp = get_error_matrix(result, gt_image_path)
        for key, val in tmp.items():
            total_results[key] += val
    relative_results['tp'] = (float(total_results['tp']) /
                              (total_results['tp'] + total_results['fn']))
    relative_results['fn'] = (float(total_results['fn']) /
                              (total_results['tp'] + total_results['fn']))
    relative_results['fp'] = (float(total_results['fp']) /
                              (total_results['fp'] + total_results['tn']))
    relative_results['tn'] = (float(total_results['tn']) /
                              (total_results['fp'] + total_results['tn']))
    logging.info("Eval results: %s", total_results)
    logging.info("Eval results relativ: %s", relative_results)
    logging.info("Positive Examples: %s ", total_results['tp'] +
                 total_results['fn'])
    logging.info("Negative Examples: %s ", total_results['fp'] +
                 total_results['tn'])
    logging.info("Accurity: %s ", float((total_results['tp']
                                        + total_results['tn'])) /
                 (total_results['tp'] + total_results['fn']
                  + total_results['fp'] + total_results['tn']))
    logging.info("%i images evaluated.", len(list_tuples))


def get_error_matrix(result, gt_image_path):
    """
    Get true positive, false positive, true negative, false negative.

    Parameters
    ----------
    result : numpy array
    gt_image_path : str
        Path to an image file with the labeled data.

    Returns
    -------
    dict
        with keys tp, tn, fp, fn
    """
    total_results = {'tp': 0,
                     'tn': 0,
                     'fp': 0,
                     'fn': 0}
    img = scipy.misc.imread(gt_image_path)
    new_img = np.zeros(img.shape)
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            new_img[i][j] = (105 == pixel)
    for gt, predict in zip(new_img.flatten(), result.flatten()):
        if gt == 0:
            if predict == 0:
                total_results['tn'] += 1
            else:
                total_results['fp'] += 1
        else:
            if predict == 0:
                total_results['fn'] += 1
            else:
                total_results['tp'] += 1
    return total_results


def scale_output(classify_image, new_shape):
    """Scale `classify_image` to `new_shape`.

    Parameters
    ----------
    classify_image : numpy array
    new_shape : tuple

    Returns
    -------
    numpy array
    """
    return scipy.misc.imresize(classify_image, new_shape, interp='nearest')


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        dest='image_path',
                        type=lambda x: utils.is_valid_file(parser, x),
                        help='load IMAGE for pixel-wise street segmenation',
                        default=utils.get_default_data_image_path(),
                        metavar='IMAGE')
    parser.add_argument('-o', '--output',
                        dest='output_path',
                        help='store semantic segmentation here',
                        default="out.png",
                        metavar='IMAGE')
    parser.add_argument('-m', '--model',
                        dest='model_path_trained',
                        help='path to the trained .caffe model file',
                        default=utils.get_model_path(),
                        metavar='MODEL')
    parser.add_argument("--stride",
                        dest="stride",
                        default=10,
                        type=int,
                        help=("the higher this value, the longer the "
                              "evaluation takes, but the more accurate it is"))
    return parser


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(image_path=args.image_path,
         output_path=args.output_path,
         model_path_trained=args.model_path_trained,
         stride=args.stride)
