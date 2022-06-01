"""
    Training curves. 
"""

import sys
sys.path.append("../../../../bin/")
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from plot_utils import smoothed, smooth_line

os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/explore/runs/tied_rnn/logdata/")

colors = list(sns.color_palette())

#%% load data

val_kappa = np.loadtxt("valKappa_tiedRNN_frozen.csv", skiprows = 1, delimiter=",")[:,2]

# val_loss = np.loadtxt("valLoss_tiedRNN_lr07.csv", skiprows = 1, delimiter=",")[:,2]

train_kappa = np.loadtxt("trainKappa_tiedRNN_frozen.csv", skiprows = 1, delimiter=",")[:,2]

# train_loss = np.loadtxt("trainLoss_tiedRNN_lr07.csv", skiprows = 1, delimiter=",")[:,2]


train_kappa_unfrozen = np.loadtxt("trainKappa_tiedRNN_unfrozen.csv", skiprows = 1, delimiter=",")[:,2]
val_kappa_unfrozen = np.loadtxt("valKappa_tiedRNN_unfrozen.csv", skiprows = 1, delimiter=",")[:,2]

steps = np.arange(len(val_kappa))

#%% kappas frozen

plt.figure(figsize=(5,5))
smooth_line(steps, val_kappa, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_kappa, weight=0.0, color=colors[1], label="Training" )
plt.hlines(0.8534, 0, 100, colors="gray", linestyles="dashed", label="base model")
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)

#%% loss frozen
# plt.figure(figsize=(5,5))
# smooth_line(steps, val_loss, weight=0.0, color=colors[0], label="Validation" )
# smooth_line(steps, train_loss, weight=0.0, color=colors[1], label="Training" )
# plt.legend(fontsize=14)
# plt.xlabel("Epochs", fontsize=16)
# plt.ylabel("Cohen's Kappa", fontsize=16)
# # plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
# plt.tick_params(labelsize=14)

#%% kappas for unfrozen

plt.figure(figsize=(5,5))
smooth_line(steps, val_kappa_unfrozen, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_kappa_unfrozen, weight=0.0, color=colors[1], label="Training" )
plt.hlines(0.870, 0, 100, colors="red", linestyles="dashed", label="constant placebo")
plt.legend(fontsize=14, loc="upper left")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)