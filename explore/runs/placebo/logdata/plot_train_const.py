"""
    Training curves. 
"""
# what plots do I need?
# Val Kappa for the various lr vals - 3 steps
# Max. val. kappa vs. lr -- interesting
# Train loss vs. steps for both ... only the best?
#
import sys
sys.path.append("../../../../bin/")
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from plot_utils import smoothed, smooth_line

os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/explore/runs/placebo/logdata/")

colors = list(sns.color_palette())

#%% load data

val_kappa_lr050 = np.loadtxt("valKappa_const_lr050.csv", skiprows = 1, delimiter=",")[:,2]
val_kappa_lr025 = np.loadtxt("valKappa_const_lr025.csv", skiprows = 1, delimiter=",")[:,2]
val_kappa_lr0125 = np.loadtxt("valKappa_const_lr0125.csv", skiprows = 1, delimiter=",")[:,2]

val_loss_lr050 = np.loadtxt("valLoss_const_lr050.csv", skiprows = 1, delimiter=",")[:,2]
val_loss_lr025 = np.loadtxt("valLoss_const_lr025.csv", skiprows = 1, delimiter=",")[:,2]
val_loss_lr0125 = np.loadtxt("valLoss_const_lr0125.csv", skiprows = 1, delimiter=",")[:,2]

train_kappa_lr050 = np.loadtxt("trainKappa_const_lr050.csv", skiprows = 1, delimiter=",")[:,2]
train_kappa_lr025 = np.loadtxt("trainKappa_const_lr025.csv", skiprows = 1, delimiter=",")[:,2]
train_kappa_lr0125 = np.loadtxt("trainKappa_const_lr0125.csv", skiprows = 1, delimiter=",")[:,2]

train_kappa_unfrozen = np.loadtxt("trainKappa_const_unfrozen_lr050.csv", skiprows = 1, delimiter=",")[:,2]
val_kappa_unfrozen = np.loadtxt("valKappa_const_unfrozen_lr050.csv", skiprows = 1, delimiter=",")[:,2]
# train_loss_lr050 = np.loadtxt("trainLoss_const_lr050.csv", skiprows = 1, delimiter=",")[:,2]
# train_loss_lr025 = np.loadtxt("trainLoss_const_lr025.csv", skiprows = 1, delimiter=",")[:,2]
# train_loss_lr0125 = np.loadtxt("trainLoss_const_lr0125.csv", skiprows = 1, delimiter=",")[:,2]

train_kappa_frozen = np.loadtxt("trainKappa_constant_frozen.csv", skiprows = 1, delimiter=",")[:,2]
val_kappa_frozen = np.loadtxt("valKappa_constant_frozen.csv", skiprows = 1, delimiter=",")[:,2]


#%% Const - Max. kappa vs. 
# BASE_NAME = "valKappa_const_lr"
# plt.figure(figsize=(5,5))
# names = ["0125","020", "025", "030", "050"]
# lrs = [0.125, 0.20, 0.25, 0.30, 0.50]
# colors = ["black"] + list(sns.color_palette())
# max_kappas = []
# for label in names:
#     kappa_table = np.loadtxt(f"{BASE_NAME}{label}.csv", skiprows = 1, delimiter=",")
#     kappa = [row[-1] for row in kappa_table]
#     max_kappas.append(np.max(kappa))

# plt.scatter(lrs, max_kappas, color="k")


#%% plot training curves LR 0.25
steps = np.arange(len(train_kappa_lr0125))

# plt.figure(figsize=(5,5))
# plt.plot(steps, train_kappa_lr025, label="training")
# plt.plot(steps, val_kappa_lr025, label="validation")
# # plt.xlim([LOW_LR, HIGH_LR])
# plt.legend(fontsize=14)
# plt.xlabel("Epochs", fontsize=16)
# plt.ylabel("Kappa", fontsize=16)
# # plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
# plt.tick_params(labelsize=14)

plt.figure(figsize=(5,5))
plt.plot(steps, train_kappa_lr050, label="LR 0.50")
plt.plot(steps, train_kappa_lr025, label="LR 0.25")
plt.plot(steps, train_kappa_lr0125, label="LR 0.125")
# plt.xlim([LOW_LR, HIGH_LR])
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Training Kappa", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)

#%%

# plt.figure(figsize=(5,5))
# plt.plot(steps, train_loss_lr050, label="LR 0.50")
# plt.plot(steps, train_loss_lr025, label="LR 0.25")
# plt.plot(steps, train_loss_lr0125, label="LR 0.125")

# # plt.plot(steps, val_loss_lr025, label="validation")
# # plt.xlim([LOW_LR, HIGH_LR])
# plt.legend(fontsize=14)
# plt.xlabel("Epochs", fontsize=16)
# plt.ylabel("Training Loss", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
# plt.tick_params(labelsize=14)

plt.figure(figsize=(5,5))
smooth_line(steps, val_loss_lr050, weight=0.8, color=colors[0], label="LR 0.50" )
smooth_line(steps, val_loss_lr025, weight=0.8, color=colors[1], label="LR 0.25" )
smooth_line(steps, val_loss_lr0125, weight=0.8, color=colors[2], label="LR 0.125" )
# plt.plot(steps, val_loss_lr050,  label="LR 0.50")
# plt.plot(steps, val_loss_lr025,  label="LR 0.25")
# plt.plot(steps, val_loss_lr0125,  label="LR 0.125")
# plt.plot(steps, val_loss_lr025, label="validation")
# plt.plot(steps, val_loss_lr025, label="validation")
# plt.xlim([LOW_LR, HIGH_LR])
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Validation Loss", fontsize=16)
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)

#%% All validation kappas

plt.figure(figsize=(5,5))
smooth_line(steps, val_kappa_lr050, weight=0.8, color=colors[0], label="LR 0.50" )
smooth_line(steps, val_kappa_lr025, weight=0.8, color=colors[1], label="LR 0.25" )
smooth_line(steps, val_kappa_lr0125, weight=0.8, color=colors[2], label="LR 0.125" )
# plt.plot(steps, val_kappa_lr050,  label="LR 0.50")
# plt.plot(steps, val_kappa_lr025,  label="LR 0.25")
# plt.plot(steps, val_kappa_lr0125,  label="LR 0.125")
# plt.xlim([LOW_LR, HIGH_LR])
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Validation Kappa", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)

#%% train kappa vs val kappa
plt.figure(figsize=(5,5))
smooth_line(steps, val_kappa_lr050, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_kappa_lr050, weight=0.0, color=colors[1], label="Training" )
plt.hlines(0.8534, 0, 100, colors="gray", linestyles="dashed", label="base model")
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)


#%% train kappa vs val kappa unfrozen
plt.figure(figsize=(5,5))
smooth_line(steps, val_kappa_unfrozen, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_kappa_unfrozen, weight=0.0, color=colors[1], label="Training" )
plt.hlines(0.8534, 0, 100, colors="gray", linestyles="dashed", label="base model")
# plt.hlines(0.870, 0, 100, colors="red", linestyles="dashed", label="constant placebo")

plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)

#%% train kappa vs val kappa frozen
plt.figure(figsize=(5,5))
smooth_line(steps, val_kappa_frozen, weight=0.0, color=colors[0], label="Validation" )
smooth_line(steps, train_kappa_frozen, weight=0.0, color=colors[1], label="Training" )
plt.hlines(0.8534, 0, 100, colors="gray", linestyles="dashed", label="base model")
# plt.hlines(0.870, 0, 100, colors="red", linestyles="dashed", label="constant placebo")

plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)