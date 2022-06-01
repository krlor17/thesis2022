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

os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/explore/runs/cnn/logdata/")

colors = list(sns.color_palette())

#%% load data

val_kappa_frozen = np.loadtxt("valKappa_cnn.csv", skiprows = 1, delimiter=",")[:,2]
val_loss_frozen = np.loadtxt("valLoss_cnn.csv", skiprows = 1, delimiter=",")[:,2]
train_kappa_frozen = np.loadtxt("trainKappa_cnn.csv", skiprows = 1, delimiter=",")[:,2]
train_loss_frozen = np.loadtxt("trainLoss_cnn.csv", skiprows = 1, delimiter=",")[:,2]

val_kappa_unfrozen = np.loadtxt("valKappa_cnn-unfrozen.csv", skiprows = 1, delimiter=",")[:,2]
val_loss_unfrozen = np.loadtxt("valLoss_cnn-unfrozen.csv", skiprows = 1, delimiter=",")[:,2]
train_kappa_unfrozen = np.loadtxt("trainKappa_cnn-unfrozen.csv", skiprows = 1, delimiter=",")[:,2]
train_loss_unfrozen = np.loadtxt("trainLoss_cnn-unfrozen.csv", skiprows = 1, delimiter=",")[:,2]


steps = np.arange(len(val_kappa_frozen))

#%% frozen

plt.figure(figsize=(5,5))
smooth_line(steps, val_kappa_frozen, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_kappa_frozen, weight=0.0, color=colors[1], label="Training" )
plt.hlines(0.8534, 0, 100, colors="gray", linestyles="dashed", label="base model")
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)

plt.figure(figsize=(5,5))
smooth_line(steps, val_loss_frozen, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_loss_frozen, weight=0.0, color=colors[1], label="Training" )
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)

#%% unfrozen

plt.figure(figsize=(5,5))
smooth_line(steps, val_kappa_unfrozen, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_kappa_unfrozen, weight=0.0, color=colors[1], label="Training" )
plt.hlines(0.870, 0, 100, colors="red", linestyles="dashed", label="constant placebo")
plt.legend(fontsize=14, loc="upper left")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)

plt.figure(figsize=(5,5))
smooth_line(steps, val_loss_unfrozen, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_loss_unfrozen, weight=0.0, color=colors[1], label="Training" )
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)