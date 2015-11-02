#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nose


def test_get_parser_test():
    from sst import test
    test.get_parser()

# def mfrdb_strip_end_test():
#     from hwrt.datasets import mfrdb
#     nose.tools.assert_equal(mfrdb.strip_end('asdf', 'df'), 'as')
