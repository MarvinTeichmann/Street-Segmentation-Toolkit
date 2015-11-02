#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Show a network.
"""

import logging
import sys
import pprint

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# Custom modules
from . import utils


def main(model_path, verbose=True):
    model, parameters = utils.deserialize_model(model_path)
    pp = pprint.PrettyPrinter(indent=4)
    print("# Model: %s" % model_path)
    if verbose:
        print("## Code")
        print(parameters['code'])
    del(parameters['code'])
    print("## Other")
    pp.pprint(parameters)


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model',
                        dest='model_path',
                        help='path to the trained .pickle model file',
                        default=utils.get_model_path(),
                        metavar='MODEL')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.model_path)
