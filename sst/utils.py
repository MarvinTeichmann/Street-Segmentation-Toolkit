#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions."""

import numpy as np
import pickle
import Image
import os
import logging
import scipy
import scipy.misc
import sys
import pkg_resources


def make_equal(features, ys):
    """Make sure the classes are equally distributed.

    Parameters
    ----------
    features : ndarray
    ys : ndarray

    Returns
    -------
    tuple of two ndarray
    """
    min_count = min(len(np.where(ys == 0)[0]), len(np.where(ys == 1)[0]))
    is_street = 0
    no_street = 0
    X_new, y_new = [], []
    for X, y in zip(features, ys):
        if y == 0 and no_street < min_count:
            no_street += 1
            X_new.append(X)
            y_new.append(y)
        elif y == 1 and is_street < min_count:
            is_street += 1
            X_new.append(X)
            y_new.append(y)
    return (np.array(X_new), np.array(y_new))


def serialize_model(model, filename, parameters=None):
    """Save a model.

    Parameters
    ----------
    model : theano expression
    filename : string
        Path where the file should be stored - should end with .pickle
    parameters : dict
        values which are relevant for the model, e.g. patch size
    """
    to_pickle = {'model': model,
                 'parameters': parameters}
    with open(filename, 'wb') as handle:
        pickle.dump(to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_model(filename):
    """Deserialize a model from a file to a Theano expression.

    Parameters
    ----------
    filename : str
        Path to a .pickle model file

    Returns
    -------
    A tuple (theano expression / nolearn model, dict of parameters)
    """
    from sst import shape
    sys.modules['shape'] = shape
    with open(filename, 'rb') as handle:
        to_pickle = pickle.load(handle)
    return to_pickle['model'], to_pickle['parameters']


def load_image_patch(filename):
    im = Image.open(filename)  # .convert('L')
    width, height = im.size
    pixels = list(im.getdata())
    features = [pixels[i * width:(i + 1) * width] for i in range(height)]
    features = np.asarray(im, dtype=np.float32).flatten()
    features /= 255.0
    return features


def load_image(filename):
    im = Image.open(filename)  # .convert('L')
    width, height = im.size
    pixels = list(im.getdata())
    features = [pixels[i * width:(i + 1) * width] for i in range(height)]
    features = np.asarray(im, dtype=np.float32)
    features /= 255.0
    return features


def overlay_images(original_image,
                   overlay,
                   output_path,
                   hard_classification=True,):
    """ Overlay original_image with segmentation_image and store the result
    with the same name as segmentation_image, but with `-overlay`.

    Parameters
    ----------
    original_image : string
        Path to an image file
    segmentation_image : string
        Path to the an image file of the same size as original_image
    hard_classification : bool
        If True, the image will only show either street or no street.
        If False, the image will show probabilities.

    Returns
    -------
    str : Path of overlay image
    """
    background = Image.open(original_image)
    overlay = scipy.misc.toimage(overlay)
    overlay = overlay.convert('RGB')

    # Replace colors of segmentation to make it easier to see
    street_color = find_street_color(overlay)
    width, height = overlay.size
    pix = overlay.load()
    pixels_debug = list(overlay.getdata())
    logging.info('%i colors in classification (min=%s, max=%s)',
                 len(list(set(pixels_debug))),
                 min(pixels_debug),
                 max(pixels_debug))
    for x in range(0, width):
        for y in range(0, height):
            if not hard_classification:
                overlay.putpixel((x, y), (0, pix[x, y][0], 0))
            else:
                if pix[x, y] == street_color:
                    overlay.putpixel((x, y), (0, 255, 0))
                else:
                    overlay.putpixel((x, y), (0, 0, 0))

    background = background.convert('RGB')
    overlay = overlay.convert('RGBA')

    # make black pixels transparent
    new_data = []
    for item in overlay.getdata():
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append((item[0], item[1], item[2], int(255 * 0.5)))
    overlay.putdata(new_data)
    background.paste(overlay, (0, 0), mask=overlay)
    background.save(output_path, 'PNG')
    return output_path


def get_overlay_name(segmentation_name):
    splitted = segmentation_name.split('.')
    splitted[-2] = splitted[-2] + '-overlay'
    output_path = '.'.join(splitted)
    return output_path


def find_street_color(im):
    """Find the color which is "street".

    Parameters
    ----------
    im : Image object opened in RGB mode

    Returns
    -------
    int tuple of length 3
    """
    # width, height = im.size
    # pix = im.load()
    # colors = {}
    # for x in range(int(0.25 * width), int(0.75 * width)):
    #     for y in range(int(0.6 * height), height):
    #         if pix[x, y] in colors:
    #             colors[pix[x, y]] += 1
    #         else:
    #             colors[pix[x, y]] = 1

    # return max(colors, key=colors.get)
    return (255, 255, 255)


def is_valid_file(parser, arg):
    """Check if arg is a valid file that already exists on the file
       system.
    """
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg
        
def filter_is_img(fname):
    
    _, ext = os.path.splitext(fname)
    
    return ext in ['.bmp', '.jpg', '.png', '.tif']


def load_color_image_features(img_path):
    """Load a colored (8-bit, RGB) image as a normalized feature vector.

    Parameters
    ----------
    img_path : string

    Returns
    -------
    numpy array
    """
    ac = scipy.misc.imread(img_path)
    ac = ac / (255.0 / 2) - 1.0
    return np.array(ac)


def sizeof_fmt(num, suffix='B'):
    """Format `num` bytes to human readable format."""
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def get_model_path():
    """
    Returns
    -------
    str
        Path to a pickle file which contains a trained model.
    """
    misc_path = pkg_resources.resource_filename('sst', 'misc/')
    return os.path.abspath(os.path.join(misc_path, 'model.pickle'))


def get_default_data_image_path():
    misc_path = pkg_resources.resource_filename('sst', 'misc/')
    return os.path.abspath(os.path.join(misc_path, "um_000000.png"))
