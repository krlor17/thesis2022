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

os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/explore/runs/alpha/logdata/")

colors = list(sns.color_palette())

#%% load data

val_kappa_alpha = np.loadtxt("valKappa_alpha_lr075.csv", skiprows = 1, delimiter=",")[:,2]
val_loss_alpha = np.loadtxt("valLoss_alpha_lr075.csv", skiprows = 1, delimiter=",")[:,2]
train_kappa_alpha = np.loadtxt("trainKappa_alpha_lr075.csv", skiprows = 1, delimiter=",")[:,2]
train_loss_alpha = np.loadtxt("trainLoss_alpha_lr075.csv", skiprows = 1, delimiter=",")[:,2]

val_kappa_simple = np.loadtxt("valKappa_simple_lr075.csv", skiprows = 1, delimiter=",")[:,2]
val_loss_simple = np.loadtxt("valLoss_simple_lr075.csv", skiprows = 1, delimiter=",")[:,2]
train_kappa_simple = np.loadtxt("trainKappa_simple_lr075.csv", skiprows = 1, delimiter=",")[:,2]
train_loss_simple = np.loadtxt("trainLoss_simple_lr075.csv", skiprows = 1, delimiter=",")[:,2]

val_kappa_unfrozen = np.loadtxt("valKappa_unfrozen_lr075.csv", skiprows = 1, delimiter=",")[:,2]
train_kappa_unfrozen = np.loadtxt("trainKappa_unfrozen_lr075.csv", skiprows = 1, delimiter=",")[:,2]
steps = np.arange(len(val_kappa_alpha))

#%% Alpha kappas

plt.figure(figsize=(5,5))
smooth_line(steps, val_kappa_alpha, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_kappa_alpha, weight=0.0, color=colors[1], label="Training" )
plt.hlines(0.8534, 0, 100, colors="gray", linestyles="dashed", label="base model")
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
plt.tick_params(labelsize=14)

#%% Alpha loss
plt.figure(figsize=(5,5))
smooth_line(steps, val_loss_alpha, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_loss_alpha, weight=0.0, color=colors[1], label="Training" )
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
plt.tick_params(labelsize=14)

#%% simple kappas

plt.figure(figsize=(5,5))
smooth_line(steps, val_kappa_simple, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_kappa_simple, weight=0.0, color=colors[1], label="Training" )
plt.hlines(0.8534, 0, 100, colors="gray", linestyles="dashed", label="base model")
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
plt.tick_params(labelsize=14)

#%% simple loss
plt.figure(figsize=(5,5))
smooth_line(steps, val_loss_simple, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_loss_simple, weight=0.0, color=colors[1], label="Training" )
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
plt.tick_params(labelsize=14)

#%% Alpha unfrozen kappas
plt.figure(figsize=(5,5))
smooth_line(steps, val_kappa_unfrozen, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_kappa_unfrozen, weight=0.0, color=colors[1], label="Training" )
plt.hlines(0.870, 0, 100, colors="red", linestyles="dashed", label="constant placebo")
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
plt.tick_params(labelsize=14)


