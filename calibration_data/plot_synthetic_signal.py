import os
import math
import numpy as np
import pandas as pd
import h5py
from scipy import signal as sp_signal
from scipy.ndimage import gaussian_filter1d
#from scipy.io import loadmat
# import mne
import matplotlib.pyplot as plt
import seaborn as sns


#%%

def metronome_signal_block(srate=500):
    arr = np.zeros(16*srate) - 0.5
    for i in range(16*srate):
        if i%(2*srate) < srate:
            arr[i] = 0.5
    return arr

def metronome_n_block(n,srate=500):
    arr = np.zeros(n*16*srate + (n-1)*3*srate)
    for i in range(n):
        arr[i*16*srate+i*3*srate:(i+1)*16*srate+i*3*srate] = metronome_signal_block(srate)
    return arr

#%%

if __name__ == "__main__":
    signal = metronome_signal_block()
    signal[0], signal[-1] = 0, 0
    t = np.linspace(0,len(signal)/500, len(signal))
    plt.figure(figsize=(5,5))
    plt.plot(t,signal, color="black")
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.xlim(-0.5,16.5)
    plt.ylim(-0.6,0.6)
    ax = plt.gca()
    ax.set_yticks([-0.5, 0, 0.5])
    plt.aspect()