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

try:
    import setuptools  # @UnusedImport # NOQA
except:
    pass

try:
    import future  # @UnusedImport # NOQA
except:
    msg = ("No module named future. Please install future first, it is needed "
           "before installing ObsPy.")
    raise ImportError(msg)

INSTALL_REQUIRES = [
    'future>=0.12.4',
    'numpy>1.4.0',
    'scipy>=0.7.2',
    ]

setup(
    # Package name
    name='pycmt3d',

    # Current version
    version='1.0',

    # A short description of package
    description='a python package for 3 dimentional centroid moment inversion',

    # A long description
    long_description=long_description,

    # Author details
    author='Wenjie Lei, Xin Song',
    author_email='lei@Princeton.EDU, songxin@physics.utoronto.ca',

    # The project's main homepage
    url='https://github.com/pypa/sampleproject',

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
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Seismology',

        # license, same with the license before
        'License :: GNU Lesser General Public License, Version 3',

        # Supproted python version
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],


    # What does your project relate to?
    keywords='centroid moment inversion',

    # Package included
    packages=find_packages(),

)
