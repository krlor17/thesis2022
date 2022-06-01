import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/explore/lrrt/alpha/logdata")
#%%

NUM_SAMPLES = 2867
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIGH_LR = 2
LOW_LR = 1e-5

lr2_table= np.loadtxt("lr_steps2.csv", skiprows = 1, delimiter=",")
lr2 = [row[-1] for row in lr2_table]

lr3_table= np.loadtxt("lr_steps3.csv", skiprows = 1, delimiter=",")
lr3 = [row[-1] for row in lr3_table]

def plot_lrrt(base_name, lr, high_lr=2, low_lr=1e-5, ):

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

    plt.xlim([low_lr, high_lr])
    plt.legend(fontsize=14)
    plt.xlabel("Learning Rate", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.tick_params(labelsize=14)

#%% simple alpha frozen

plot_lrrt("trainLoss_simple_wd", lr3, high_lr = 3)


#%% simple alpha unfrozen

plot_lrrt("trainLoss_simple_unfrozen_wd", lr3, high_lr = 3)

#%% alpha frozen

plot_lrrt("trainLoss_alpha_wd", lr2, high_lr = 2)

#%% 

plot_lrrt("trainLoss_alpha_unfrozen_wd", lr2, high_lr = 2)
