#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:33:45 2022

@author: kafkan
"""
import os
from math import ceil, floor
import numpy as np
import h5py
from scipy import signal
# from scipy.io import loadmat
import scipy.signal as sp_signal
import mne
import matplotlib.pyplot as plt
import sklearn.cluster as skc
import seaborn as sns
import time as systime
import pandas as pd


def to_label_dict(arr):
    num_channels = max(arr.shape)
    s_length = min(arr.shape)
    # invert shape
    strings = dict()
    for i in range(num_channels):
        s = ""
        for j in range(s_length):
            s += chr(arr[j,i])
        # remove trailing whitespace
        s = s.strip()
        strings[s] = i
    return strings

filename = "/media/kafkan/storage1/speciale/data/study_1A_mat_simple/S_01/night_1/EEG_raw.mat"

with h5py.File(filename, 'r') as f:
    chanlabels, data, srate = f["chanlabels"], f['data'], f['srate'][0,0]
    channels = to_label_dict(chanlabels[:])

    half_hour_steps = int(30*60*srate)
    
    # clear eye activity - calibration end!
    from_t1 = 405*500
    to_t1 = 485*500
    
    from_t2 = to_t1 - 189*500
    to_t2 = from_t2  + 35*500
    channel = "EOGr"
    # channel = "EOGl"

    end = data[from_t1:to_t1,channels[channel]]
    start =  data[from_t2:to_t2,channels[channel]]
    
    time_end = np.linspace(from_t1/500, to_t1/500, len(end))
    time_start = np.linspace(from_t2/500, to_t2/500, len(start))
    
    plt.figure(figsize=(7,5))
    plt.plot(time_start, start/1000, color="k")
    plt.xlabel("time [s]", fontsize=18)
    plt.ylabel("EOGr $-$ avg. [mV]", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.figure(figsize=(7,5))
    plt.plot(time_end, end/1000, color="k")
    plt.xlabel("time [s]", fontsize=18)
    plt.ylabel("EOGr  $-$ avg. [mV]", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)