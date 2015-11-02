#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate the error (tp, tn, fp, fp) on all images.
"""

import logging
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# Custom modules
from . import utils
from . import eval_image
from . import view


def main(model_path_trained, stride):
    trained, paramters = utils.deserialize_model(model_path_trained)
    view.main(model_path_trained, verbose=False)
    eval_image.eval_pickle(trained, paramters, 'testing.pickle', stride=stride)


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model',
                        dest='model_path_trained',
                        help='path to the trained .pickle model file',
                        default=utils.get_model_path(),
                        metavar='MODEL')
    parser.add_argument("--stride",
                        dest="stride",
                        default=10,
                        type=int,
                        help=("the higher this value, the longer the "
                              "evaluation takes, but the more accurate it is"))
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.model_path_trained,
         args.stride)
