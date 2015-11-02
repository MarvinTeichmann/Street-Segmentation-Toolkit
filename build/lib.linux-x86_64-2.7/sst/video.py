#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a video. (TODO: better description)
"""

import logging
import sys
import os

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# Custom modules
from . import utils
from . import eval_image


def make_video(model_path_trained, photo_folder, video_dir, stride):
    """
    Parameters
    ----------
    model_path_trained : str
        Use this model for evaluation
    photo_folder : str
        Folder with original images.
    video_dir : str
        Write overlayed images in this folder.
    stride : int
    """
    files_data = [f for f in sorted(os.listdir(photo_folder))
                  if f.endswith('.png')]
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    nn, parameters = utils.deserialize_model(model_path_trained)
    for i, image_name in enumerate(files_data):
        logging.info("Processing image %s / %s", (i + 1), len(files_data))
        image_path = os.path.join(photo_folder, image_name)
        result = eval_image.eval_net(nn,
                                     image_path,
                                     parameters,
                                     stride=stride)
        utils.overlay_images(image_path,
                             result,
                             os.path.join(video_dir, "%04d.png" % i))
    cmd = "avconv -f image2 -i %s/%%04d.png avconv_out.avi" % video_dir
    logging.info(cmd)
    os.system(cmd)


def main(model_path_trained, photo_folder, video_dir, stride):
    make_video(model_path_trained, photo_folder, video_dir, stride)


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model',
                        dest='model_path_trained',
                        help='path to the trained .pickle model file',
                        default=utils.get_model_path(),
                        metavar='MODEL')
    parser.add_argument("--video-dir",
                        dest="video_dir",
                        default=('video'),
                        help="directory where overlay images will be put")
    parser.add_argument("--photo-dir",
                        dest="photo_dir",
                        default=('/data/ml-prak_daten/'
                                 'segmentation_data/dataset1'),
                        help="directory with data images")
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
         args.photo_dir,
         args.video_dir,
         args.stride)
