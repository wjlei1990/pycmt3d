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


def gen_random_array(npts, sample_number=0):

    if npts <= 0:
        return
    if sample_number <= 1:
        sample_number = 1
    subset_array = np.zeros(npts)
    location_array = np.random.choice(npts, sample_number)
    for _idx in location_array:
        subset_array[_idx] += 1
    return subset_array