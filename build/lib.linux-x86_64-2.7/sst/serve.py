#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Start a webserver which can record the data and work as a classifier."""

import logging
import scipy
import sys
import time
import uuid

from os.path import basename, join as pjoin
from flask import Flask, request, render_template, url_for
from flask_bootstrap import Bootstrap
from pkg_resources import resource_filename

try:
    from urllib import parse as urlquote
except ImportError:  # Python 2 fallback
    from urllib import quote as urlquote

# Custom modules
from . import utils
from .utils import overlay_images, get_overlay_name
from .eval_image import eval_net

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

# configuration
DEBUG = True


template_path = resource_filename('sst', 'templates/')
app = Flask(__name__,
            template_folder=template_path,
            static_path=resource_filename('sst', 'static/'))
Bootstrap(app)
app.config.from_object(__name__)
nn = None
nn_params = None


@app.route('/', methods=['POST', 'GET'])
def index():
    """Start page."""
    return (('<a href="work?photo_path=%s&stride=100">'
             'Classify </a>') %
            urlquote(utils.get_default_data_image_path()))


@app.route('/work', methods=['POST', 'GET'])
def work():
    """a worker task"""
    global nn
    if request.method == 'GET':
        out_filename = pjoin(resource_filename('sst', 'static/'),
                             'out-%s.png' % uuid.uuid4())

        photo_path = request.args.get('photo_path',
                                      utils.get_default_data_image_path())
        patch_size = nn_params['patch_size']
        hard_classification = request.args.get('hard_classification',
                                               '0') == '1'
        output_path = request.args.get('output_path', out_filename)
        stride = min(int(request.args.get('stride', 10)), patch_size)
        logging.info("photo_path: %s", photo_path)
        t0 = time.time()
        logging.info('photo_path: %s', photo_path)
        logging.info('parameters: %s', nn_params)
        result = eval_net(trained=nn,
                          photo_path=photo_path,
                          parameters=nn_params,
                          stride=stride,
                          hard_classification=hard_classification)
        scipy.misc.imsave(output_path, result)
        t1 = time.time()
        overlay_path = overlay_images(photo_path,
                                      result,
                                      get_overlay_name(output_path))
        logging.info("output_path: %s", output_path)
        logging.info("Overlay path: %s", overlay_path)
        t2 = time.time()
        output_path = url_for('static',
                              filename=basename(output_path))
        tmp = basename(get_overlay_name(output_path))
        output_overlay_path = url_for('static',
                                      filename=tmp)
        return render_template('canvas.html',
                               execution_time=t1 - t0,
                               overlay_time=t2 - t1,
                               photo_path=photo_path,
                               patch_size=patch_size,
                               stride=stride,
                               output_path=output_path,
                               output_overlay_path=output_overlay_path,
                               hard_classification=hard_classification)
    else:
        return "Request method: %s" % request.method


def get_parser():
    """Return the parser object for this script."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model',
                        dest='model_path_trained',
                        help='path to the trained .caffe model file',
                        default=utils.get_model_path(),
                        metavar='MODEL')
    parser.add_argument("--port",
                        dest="port", default=5000, type=int,
                        help="where should the webserver run")
    return parser


def main(port, model_path_trained):
    """Main function starting the webserver."""
    global nn, nn_params, template_path
    logging.info('template_path: %s', template_path)
    if nn is None:
        nn, nn_params = utils.deserialize_model(model_path_trained)
    logging.info("Start webserver...")
    app.run(port=port)

if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.port, args.model_path_trained)
