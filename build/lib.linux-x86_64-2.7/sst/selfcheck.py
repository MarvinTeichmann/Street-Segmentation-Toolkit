#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Check if all necessary modules / programs / files for sst are there and
   if the version is ok.
"""

import imp
import sys
import platform
import os
import pkg_resources


class Bcolors(object):
    """Terminal colors with ANSI escape codes."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def which(program):
    """Get the path of a program or ``None`` if ``program`` is not in path."""
    def is_exe(fpath):
        """Check for windows users."""
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def check_python_version():
    """Check if the currently running Python version is new enough."""
    # Required due to multiple with statements on one line
    req_version = (2, 7)
    cur_version = sys.version_info
    if cur_version >= req_version:
        print("Python version... %sOK%s (found %s, requires %s)" %
              (Bcolors.OKGREEN, Bcolors.ENDC, str(platform.python_version()),
               str(req_version[0]) + "." + str(req_version[1])))
    else:
        print("Python version... %sFAIL%s (found %s, requires %s)" %
              (Bcolors.FAIL, Bcolors.ENDC, str(cur_version),
               str(req_version)))


def check_python_modules():
    """Check if all necessary / recommended modules are installed."""
    print("\033[1mCheck modules\033[0m")
    required_modules = ['argparse', 'theano', 'numpy', 'lasagne',
                        'nolearn', 'nose', 'nose-cov']
    found = []
    for required_module in required_modules:
        try:
            imp.find_module(required_module)
            check = "module '%s' ... %sfound%s" % (required_module,
                                                   Bcolors.OKGREEN,
                                                   Bcolors.ENDC)
            print(check)
            found.append(required_module)
        except ImportError:
            print("module '%s' ... %sNOT%s found" % (required_module,
                                                     Bcolors.WARNING,
                                                     Bcolors.ENDC))

    if "argparse" in found:
        import argparse
        print("argparse version: %s (1.1 tested)" % argparse.__version__)
    if "theano" in found:
        try:
            import theano
            print("theano version: %s (0.7.0 tested)" %
                  theano.__version__)
        except RuntimeError:
            print(("theano could %sNOT%s be imported. It is most likely that "
                   "you configured your GPU the right way, but another task "
                   "consumes too much of its memory.") %
                  (Bcolors.FAIL, Bcolors.ENDC))
    if "numpy" in found:
        import numpy
        print("numpy version: %s (1.8.2 tested)" %
              numpy.__version__)
    if "lasagne" in found:
        try:
            import lasagne
            print("lasagne version: %s (0.1 dev tested)" %
                  lasagne.__version__)
        except RuntimeError:
            print(("lasagne could %sNOT%s be imported. It is most likely that "
                   "you configured your GPU the right way, but another task "
                   "consumes too much of its memory.") %
                  (Bcolors.FAIL, Bcolors.ENDC))
        except AttributeError as e:
            print("Strange %sAttributeError%s for lasagne: %s" %
                  (Bcolors.FAIL, Bcolors.ENDC, str(e)))


def check_executables():
    """Check if all necessary / recommended executables are installed."""
    print("\033[1mCheck executables\033[0m")
    required_executables = ['nvcc']
    for executable in required_executables:
        path = which(executable)
        if path is None:
            print("%s ... %sNOT%s found" % (executable, Bcolors.WARNING,
                                            Bcolors.ENDC))
        else:
            print("%s ... %sfound%s at %s" % (executable, Bcolors.OKGREEN,
                                              Bcolors.ENDC, path))


def main():
    """Execute all checks."""
    check_python_version()
    check_python_modules()
    check_executables()
    print("\033[1mCheck environment variables\033[0m")
    if 'DATA_PATH' in os.environ:
        print('DATA_PATH: %s' % os.environ['DATA_PATH'])
    else:
        print('DATA_PATH is not set')
    # home = os.path.expanduser("~")
    print("\033[1mCheck files\033[0m")
    misc_path = pkg_resources.resource_filename('sst', 'misc/')
    print("misc-path: %s" % misc_path)


if __name__ == '__main__':
    main()
