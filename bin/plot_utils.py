#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities

@author: krlor17
"""

import numpy as np
import matplotlib.pyplot as plt

def smoothed(data, weight):
    """
    Produces a smoothed version of the data using an exp. moving avg.

    Parameters
    ----------
    data : any list or array accepted by np.array
        1D array of values
    weight : float
        A value between 0 and 1 giving the weight of the smoothing

    Returns
    -------
    A numpy array of the smoothed data

    """
    assert 0 <= weight <= 1
    data = np.array(data, copy=True)
    mwa = np.empty_like(data)
    prev = data[0]
    for i, val in enumerate(data):
        mwa[i] = (1-weight)*val + weight*prev
        prev = mwa[i]
    return mwa

def smooth_line(x,data, color, weight=0.5 ,label=None, alpha=0.25):
    plt.plot(x, smoothed(data, weight), color=color, label=label)
    plt.plot(x, data, color=color, alpha=alpha)
        