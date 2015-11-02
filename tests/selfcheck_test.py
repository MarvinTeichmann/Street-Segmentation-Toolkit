#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nose


def check_python_version_test():
    from sst import selfcheck
    selfcheck.check_python_version()

# def mfrdb_strip_end_test():
#     from hwrt.datasets import mfrdb
#     nose.tools.assert_equal(mfrdb.strip_end('asdf', 'df'), 'as')
