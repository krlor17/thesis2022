import os
import math
import numpy as np
import scipy.signal as signal
import h5py
import mne
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json


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

def butter_bandpass(data,lowcut, highcut, srate=500, order=3):
    nyquist = 0.5*srate
    low = lowcut/nyquist
    high = highcut/nyquist
    b,a = signal.butter(order, [low,high], btype="band")
    return signal.lfilter(b,a,data)

def butter_highpass(data,cut, srate=500, order=3):
    nyquist = 0.5*srate
    cutoff = cut/nyquist
    b,a = signal.butter(order, cutoff, btype="high")
    return signal.filtfilt(b,a,data)

def butter_lowpass(data,cut, srate=500, order=3):
    nyquist = 0.5*srate
    cutoff = cut/nyquist
    b,a = signal.butter(order, cutoff, btype="low")
    return signal.filtfilt(b,a,data)

BAD_CHAN_FILE = "/media/kafkan/storage1/speciale/mne/bad_channels.json"

BASE = "/media/kafkan/storage1/speciale/data/study_1A_mat_simple/S_{s}/night_{n}/"
MAT = "EEG_raw.mat"
CALRAW = "calibrationData/rawEEG.npy"
CALSPEC = "calibrationData/calSpectogram.npy"
CALMATRIX = "calibrationData/foldedMatrix.npy"
SRATE = 500             # sampling rate in Hz
CAL_LENGTH = 189        # calibration length in seconds

EAR_CHANNELS = ["ELA","ELB","EL"]

if __name__ == "__main__":
    
    with open(BAD_CHAN_FILE,"r") as js_file:
        bad_channels = json.load(js_file)
    
    times = pd.read_csv("/media/kafkan/storage1/speciale/peak_detection_tests/end_test_hour/times.csv")
    times["S"] = times.S.astype(int).astype(str)
    times["S"] = np.array([s.zfill(2) for s in times.S])
    times["N"] = times.N.astype(int).astype(str)
    cnt = 0
    
    for _, values in tqdm(times.iterrows()):
            s, n, _, end, _ = values
            end = int(end)
            base = BASE.format(s=s, n=n)
            in_file = h5py.File(BASE.format(s=s,n=n)+MAT)

            channels = to_label_dict(in_file["chanlabels"][:])
            
            raw = in_file["data"][end*SRATE-CAL_LENGTH*SRATE:end*SRATE,:]
            
            ### PREPROCESSING
            
            raw[np.isnan(raw)] = np.nanmean(raw) ## good idea ?
            cal = butter_lowpass(raw, cut=100, srate=SRATE) # anti-aliasing filter, cut at 100 Hz
            cal = butter_highpass(cal, cut=0.1, srate=100)
            cal = cal[::5,:] # "decimate" / downsample to 100 Hz
            
        
            bad = bad_channels[f"S{int(s)}N{n}"]
            
            chan_mask = [True]*12 + [False]*13 # keep the 12 Ear-EEG channels, unless bad
            for ch in bad:
                chan_mask[channels[ch]] = False        
        
        
            new_avg = np.nanmean(cal[:,chan_mask], axis=1)
            for c in range(0,12):
                cal[:,c] = cal[:,c] - new_avg
            
            cal = cal[:,0:12]
            
            # cal = cal - np.mean(cal)
            cal = cal - np.nanmean(cal,axis=(0)) ## pr channel
            cal = np.clip(cal, a_min = -400, a_max = 400)
            ### 
            
            # folded matrix
            depth = raw.shape[1]
            n_cols = math.ceil(CAL_LENGTH/30)
            n_rows = 30*SRATE
            matrix = np.zeros((n_rows,n_cols,depth))
            for i in range(n_cols):
                remainder = min(len(raw)-i*n_rows, n_rows)
                matrix[0:remainder,i,:] = raw[i*n_rows :i*n_rows + remainder,:]
            
            ## down = raw[::5,:] # downsampled 5 times!
            ## nanmean = np.nanmean(down)
            ## if nanmean != np.nan:
            ##     down[np.isnan(down)] = np.nanmean(down)
            ## else:
            ##     down[np.isnan(down)] = 0
            
            # STFT as in Phan et al. 2019 (SeqSleepNet): 2 second hamming window w. 50% overlap and 256 point FFT
            ### obs. downsampled to 100 Hz
            ###  OBS. set to a 2 second window
            # spectrogram = np.zeros((129,214,25))
            spectrogram = np.zeros((129,107,25))

            # f,t,spectrogram = sps.stft(down[:,10], fs=100, window="hamming", nperseg=200, nfft=256)
            
            # stop = False
            # for i in range(25):
            #     # try:
            #     _, _, spec = signal.spectrogram(down[:,i], fs=100, window="hamming", nperseg=2*100, nfft=256)
            #     spectrogram[:,:,i] = spec
            #     # except:
            #     #     stop = True
            #     #     print(s,n,i)
            #     # if stop:
            #     #     raise SystemExit(0)
            #     # plt.figure()
            #     # plt.imshow(spectrogram[:,:,i])
            LEFT = [channels[ch] for ch in channels.keys() if "EL" in ch]
            RIGHT = [channels[ch] for ch in channels.keys() if "ER" in ch]
            diff = np.mean(cal[:,RIGHT], axis=1) - np.mean(cal[:,LEFT], axis=1)
            _, _, spectrogram = signal.spectrogram(diff, fs=100, window="hamming", nperseg=2*100, noverlap=100, nfft=256)
            
            plt.figure()
            plt.imshow(spectrogram)
            

            #_,_,spectrogram = sps.spectrogram(down[:,10], fs=100, window="hamming", nperseg=200, nfft=256)
            # _,_,spectrogram = sps.spectrogram(down[:,:], fs=100, window="hamming", nperseg=100, nfft=256, axis=0)
            # plt.figure()
            # plt.plot(raw[:,10])
            # np.save(base+CALMATRIX, matrix)
            # np.save(base+CALRAW, raw)
            # np.save(base+CALSPEC, spectrogram)
            in_file.close()
            cnt += 1
            # if cnt > 3:
            #     raise SystemExit(0)


