import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/extending_recurrent/lrrt/alpha/logdata")
#%%

NUM_SAMPLES = 2867
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIGH_LR = 2
LOW_LR = 1e-5

lr2_table= np.loadtxt("lr2.csv", skiprows = 1, delimiter=",")
lr2 = [row[-1] for row in lr2_table]

def plot_lrrt(base_name, lr, high_lr=2, low_lr=1e-5, lim=None, loc = "best" ):

    plt.figure(figsize=(6,5))
    name = ["0", "1e-4", "1e-3"]
    value = ["0.00", "1e-4", "1e-3"]
    colors = ["black"] + list(sns.color_palette())
    colors.pop(3)
    for wd, label, color in zip(name, value, colors):
        print(f"{base_name}{wd}.csv")
        loss_table = np.loadtxt(f"{base_name}{wd}.csv", skiprows = 1, delimiter=",")
        loss = [row[-1] for row in loss_table]
        plt.plot(lr, loss, label=f"Weight Decay = {label}", color=color)
    if lim:
        plt.ylim(lim)
    plt.xlim([low_lr, high_lr])
    plt.legend(fontsize=14, loc=loc)
    plt.xlabel("Learning Rate", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.tick_params(labelsize=14)

#%% constant placebo epochwise

plot_lrrt("trainLoss_alpha_epoch_wd", lr2, high_lr = 2, lim=[6e-4,0.002])


#%% constant placebo unfrozen

plot_lrrt("trainLoss_alpha_sequence_wd", lr2, high_lr = 2, loc="upper left")

