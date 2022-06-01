#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 20:12:00 2022

@author: kafkan
"""
import os
import math
import numpy as np
import pandas as pd
import h5py
from scipy import signal as sp_signal
from scipy.ndimage import gaussian_filter1d
import mne
import matplotlib.pyplot as plt
import sys
import traceback
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
import seaborn as sns
from tqdm import tqdm

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

def butter_highpass(data,cut, srate=500, order=3):
    nyquist = 0.5*srate
    cutoff = cut/nyquist
    b,a = sp_signal.butter(order, cutoff, btype="high")
    return sp_signal.filtfilt(b,a,data)


#%%

MV_AVG_SEC = 16 # 8 or 16
MAX_ANGLE = 85
def get_calibration_end(fpath, signal, srate=500, window_len = 30*60, debug=False):
    """
    Find the end of the calibration sequence in the data contained in the .mat file at fpath

    Parameters
    ----------
    fpath : str
        path to .mat file.
    signal : numpy.Array
        Array with the signal used to locate end of calibration.
    srate : int, optional
        Sampling rate of signal i.e. number of points pr. second. The default is 500.
    window_len : int, optional
        The time from start to look for calibration end in seconds. The default is 30*60.
    debug : bool, optional
        Show debug plots and text. The default is False.

    Returns
    -------
    start: float
        Start point of calibration sequence in seconds
    end: float
        End point of calibration sequence in seconds

    """
    
    with h5py.File(fpath, 'r') as f:
        chanlabels, data, srate = f["chanlabels"], f['data'], int(f['srate'][0,0])
        channels = to_label_dict(chanlabels[:])
        
        # compile channel statistics (nan and std.)
        stats = []
        window_all = data[0:window_len*srate,:]
        for key in channels.keys():
            std = np.std(window_all[:,channels[key]])
            nan_sum = np.sum(np.isnan(window_all[:,channels[key]]))
            stats.append([key, std, nan_sum])
        del window_all
        stats = pd.DataFrame(stats, columns=["Channel", "Std", "Num. Nan"])
        
        # filter channels based on std and whether there is nan values in the channels
        chan_mask = ((stats.Std.isna() | (stats.Std > 2*stats.mean()["Std"]))).to_numpy()
        chan_mask = np.invert(chan_mask)
        chan_mask[channels["EOGl"]] = True
        chan_mask[channels["EMGl"]] = False
        chan_mask[channels["EMGr"]] = False
        chan_mask[channels["EMGc"]] = False  
        
        # choose between EOGr and EOGl channel for analysis
        channel = "EOGr" if not chan_mask[channels["EOGr"]] else "EOGl"
        
        series = data[0:window_len*srate,channels[channel]]
        
        series[np.isnan(series)] = np.nanmean(series)
        
        # remove filtered channels and rereference
        new_avg = np.mean(np.nan_to_num(data[0:window_len*srate,chan_mask]), axis=1)
        series -= new_avg
    
    # high-pass filter
    series = butter_highpass(series, 0.1, srate=srate)
    
    # center to mean
    series -= np.mean(series)
    
    # clip input
    series = np.clip(series, -100,100)
    
    # correlated with signal
    series = np.abs(np.correlate(series, signal, mode="same"))
    
    # get start, end from series
    start, end = mine_signal_match(series, srate, debug=debug)
    
    
    return start, end
    
NUM_POINTS = 20
DOWNSAMPLING_RATE = 200
NEIGHBORS = 20
N_STD = 2

def mine_signal_match(series, srate, debug=False):

    
    # cumulated correlated series
    cumulated = np.cumsum(series)
    
    # downsampling
    cumulated = cumulated[::DOWNSAMPLING_RATE]
    
    ### Mine matches as peaks
    optics = OPTICS(min_samples=NEIGHBORS)
    clustering = optics.fit(cumulated.reshape(-1,1))
    reach = clustering.reachability_
    core = clustering.core_distances_
    order = clustering.ordering_
    
    # std deviation
    reach_copy = reach.copy()
    reach_copy[reach == np.inf] = 0
    std = np.std(reach_copy)
    mean = np.mean(reach_copy)
    
    preds = cluster_optics_dbscan(reachability=reach, core_distances=core, ordering=order, eps = mean + N_STD*std)

    
    if debug:
        print("labels:",np.unique(preds))
        print(sum(reach == np.inf))
        print("std. dev.:",std)
        plt.figure()
        plt.plot(np.linspace(0,30*60,len(reach)),reach)
        plt.hlines(y=mean + N_STD*std, xmin =0, xmax = 30*60, colors = "red")
        
        plt.figure()
        sns.scatterplot(x=np.linspace(0,1,len(cumulated)), y=cumulated, hue=preds, palette="deep", s=2)
    
    peaks = []
    prev_label = None
    current = None
    last_idx = len(preds) -1
    for i, label in enumerate(preds):
        if prev_label != label or i == last_idx:
            if prev_label == -1:
               current.set_end(i)
               peaks.append(current) 
            elif label == -1:
                current = Peak(i, srate, DOWNSAMPLING_RATE)

            prev_label = label
                
    if debug:
        print("Number of peaks found:",len(peaks))
        print("peak durations:\t", end="")
        for peak in peaks:
            print(peak.duration(), end="\t")

    if len(peaks) == 1:
        peak = peaks[0]
        return peak.get_start(), peak.get_end()
    
    times = [peak.get_start() for peak in peaks]
    volumes = [peak.duration() for peak in peaks]
    
    ## assign score based on order statistics
    # ideally close to bed time i.e. small time best: descending order
    time_order = np.argsort(times)[::-1]

    # high volume best, sort in ascending order
    vol_order = np.argsort(volumes)
    
    num_peaks = len(peaks)
    scores = np.zeros(num_peaks)
    for i, orders in enumerate(zip(time_order, vol_order)):
        t_idx, v_idx = orders
        scores[t_idx] = 0.2*(1/num_peaks + i*1/num_peaks)
        scores[v_idx] = 1/num_peaks + i*1/num_peaks
    
    best = peaks[np.argmax(scores)]
    return best.get_start(), best.get_end()

class Peak:
    def __init__(self, start_idx, srate, downsampling):
        self.start = start_idx
        self.srate = srate
        self.ds = downsampling
        self.end = -1
        
    def set_end(self, idx):
        self.end = idx

    def get_end(self):
        return self.end/self.srate*self.ds
    
    def get_start(self):
        return self.start/self.srate*self.ds
    
    def duration(self):
        assert self.end > 0
        return (self.end - self.start)/self.srate*self.ds

#%%

# if __name__ == '__main__':
#     signal = metronome_signal_block()
#     filename = "EEG_raw.mat"
#     plt.ioff()
#     columns=["S","N","start","end","srate"]
#     times = pd.DataFrame(columns=columns)
#     srate = 500
#     for sub in tqdm(range(1,20+1)):
#         s = str(sub)
#         if len(s)<2:
#             s = "0"+s
#         for night in range(1,4+1):
#             directory = "/media/kafkan/storage1/speciale/data/study_1A_mat_simple/S_"+s+"/night_"+str(night)+"/"
#             try:
#                 plt.ioff()
#                 start, end= get_calibration_end(directory+filename, signal)
#                 times = times.append(pd.DataFrame([[s,night,start,end,srate]], columns=columns))
                
#             except ValueError:
#                 print(f"ValueError on S{s} N{night}")
#                 traceback.print_exception(*sys.exc_info())
#             except OSError:
#                 print(f"OsError on S{s} N{night}")
#                 traceback.print_exception(*sys.exc_info())
#             except:
#                 print(f"Error on S{s} N{night}")
#     times.to_csv("/media/kafkan/storage1/speciale/peak_detection_tests/end_test_reref/times.csv", index=False)
                
#     for sub in tqdm(range(1,20+1)):
#         s = str(sub)
#         if len(s)<2:
#             s = "0"+s
#         for night in range(1,4+1):
#             directory = "/media/kafkan/storage1/speciale/data/study_1A_mat_simple/S_"+s+"/night_"+str(night)+"/"
#             try:
#                 plt.ioff()
#                 with h5py.File(directory + filename, 'r') as f:
#                     chanlabels, data, srate = f["chanlabels"], f['data'], f['srate'][0,0]
#                     channels = to_label_dict(chanlabels[:])
#                     row = times[(times.S == s) & (times.N == night)]
#                     start, end = row["start"], row["end"]

#                     match = data[int(start)*500:int(end)*500,channels["EOGr"]]
#                     time = np.linspace(start,end,len(match))
#                     fig=plt.figure()
#                     plt.plot(time, match)
#                     fig.savefig("/media/kafkan/storage1/speciale/peak_detection_tests/end_test_reref/S"+s+"N"+str(night))
#             except ValueError:
#                 print(f"ValueError on S{s} N{night}")
#                 traceback.print_exception(*sys.exc_info())
#             except OSError:
#                 print(f"OsError on S{s} N{night}")
#                 traceback.print_exception(*sys.exc_info())
#             except:
#                 print(f"Error on S{s} N{night}")
#                 traceback.print_exception(*sys.exc_info())

    
    
#%%
if __name__ == '__main__':
    #signal = metronome_signal_block()
    signal = metronome_n_block(1)
    # signal = np.load("/home/kafkan/git-repos/speciale/extract_calibration/mean_signal.npy")[40:380]
    directory = "/media/kafkan/storage1/speciale/data/study_1A_mat_simple/S_17/night_1/"
    directory = "/media/kafkan/storage1/speciale/data/study_1A_mat_simple/S_13/night_4/"
    #filename = "/media/kafkan/storage1/speciale/data/study_1A_mat_simple/S_18/night_4/EEG_raw.mat"
    filename = "EEG_raw.mat"
    
    start, end = get_calibration_end(directory+filename, signal, debug=True)
    print("Start:",start,"\t End:",end)
    with h5py.File(directory+filename, 'r') as f:
        chanlabels, data, srate = f["chanlabels"], f['data'], f['srate'][0,0]
        channels = to_label_dict(chanlabels[:])
        #plt.figure()
        # start, end, n = get_calibration_end([data[:,channels["EOGr"]]], signal, bed_time, angle_thresh=1.5, debug=True)
        series = data[0:30*60*500,channels["EOGr"]]
        match = series[int(start)*500 : int(end)*500]
        time = np.linspace(start,end, len(match))
        plt.figure()
        plt.plot(time, match)