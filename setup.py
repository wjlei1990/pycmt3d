#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
pycmt3d - a Python package for 3-dimenional centroid moment inversion

:copyright:
    Wenjie Lei (lei@Princeton.EDU), 2015
    Xin Song (songxin@physics.utoronto.ca), 2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lgpl.html)
'''
# Importing setuptools monkeypatches some of distutils commands so things like
# 'python setup.py develop' work. Wrap in try/except so it is not an actual
# dependency. Inplace installation with pip works also without importing
# setuptools.

import os
import glob
from setuptools import setup
from setuptools import find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except Exception as e:
        return "Can't open %s" % fname

INSTALL_REQUIRES = [
    'future>=0.12.4',
    'numpy>1.4.0',
    ]

long_description = """
Source code: https://github.com/wjlei1990/pycmt3d

Documentation: http://wjlei1990.github.io/pycmt3d/

%s""".strip() % read("README.md")

setup(
    # Package name
    name='pycmt3d',

    # Current version
    version='0.1.0',

    # A short description of package
    description='a python port of cmt3d softward for 3 dimensional centroid moment tensor inversion',

    # A long description
    long_description=read("README.md"),

    # Author details
    author='Wenjie Lei, Xin Song',
    author_email='lei@Princeton.EDU, songxin@physics.utoronto.ca',

    # The project's main homepage
    url='https://github.com/wjlei1990/pycmt3d',

    # The license
    license='GNU Lesser General Public License, Version 3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # The project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        # Supproted python version
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],

    # Package included
    packages=find_packages("src"), requires=['numpy', 'obspy', 'obspy'],
    package_dir={"": "src"},
    py_modules=[os.path.splitext(os.path.basename(i))[0]
                for i in glob.glob("src/*.py")],

    # What does your project relate to?
    keywords=['seismology', 'cmt3d', 'moment tensor', 'centroid moment inversion'],
    install_requires=[
        "obspy", "numpy", "future>=0.14.1"
    ],
    extras_require={
        "docs": ["sphinx"]
    }
)
