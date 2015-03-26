#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

def sum_matrix(coef, data):
    if len(coef) != len(data):
        raise ValueError("dimension of coef and data not the same")
    sum = coef[0] * data[0]
    for _idx in range(1, len(coef)):
        sum += coef[_idx] * data[_idx]

    return sum

def gen_random_array(npts, threshold=0):

    if npts <= 0:
        return
    sum = 0
    while sum <= threshold:
        random_array = np.random.randint(2, size=npts)
        sum = sum(random_array)

    return random_array