#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys


def uni(x):
    if sys.version_info >= (3,):
        return x.decode()
    else:
        return x
