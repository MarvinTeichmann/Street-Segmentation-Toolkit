#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nose
import os
import pkg_resources
import numpy as np
import scipy

# tested module
from sst import eval_image


def eval_script_get_parser_test():
    eval_image.get_parser()


def get_error_matrix_test():
    misc_path = pkg_resources.resource_filename('sst', 'misc/')
    gt_image_path = os.path.abspath(os.path.join(misc_path,
                                                 "uu_road_000000.png"))
    img = scipy.misc.imread(gt_image_path)
    prediction = np.zeros(img.shape)
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            prediction[i][j] = (105 == pixel)
    returned = eval_image.get_error_matrix(prediction, gt_image_path)
    nose.tools.assert_equal({'fn': 0, 'fp': 0, 'tn': 99030, 'tp': 17718},
                            returned)

    # make all false
    for i, row in enumerate(prediction):
        for j, pixel in enumerate(row):
            prediction[i][j] = 1 - prediction[i][j]  # swap values
    returned = eval_image.get_error_matrix(prediction, gt_image_path)
    nose.tools.assert_equal({'fn': 17718, 'fp': 99030, 'tn': 0, 'tp': 0},
                            returned)


# def mfrdb_strip_end_test():
#     from hwrt.datasets import mfrdb
#     nose.tools.assert_equal(mfrdb.strip_end('asdf', 'df'), 'as')
