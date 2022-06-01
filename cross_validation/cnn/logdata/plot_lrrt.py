import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/cross_validation/cnn/logdata")
#%%

NUM_SAMPLES = 2867
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIGH_LR = 1.6
LOW_LR = 1e-5


BASE_NAME = "trainLoss_cnn_wd"

lr_table= np.loadtxt("lr_steps.csv", skiprows = 1, delimiter=",")
lr = [row[-1] for row in lr_table]
plt.figure(figsize=(6,5))
name = ["0", "1e-4", "1e-3"]
value = ["0.00","1e-4", "1e-3"]
colors = ["black"] + list(sns.color_palette())
colors.pop(3)
for wd, label, color in zip(name, value, colors):
    print(f"{BASE_NAME}{wd}.csv")
    loss_table = np.loadtxt(f"{BASE_NAME}{wd}.csv", skiprows = 1, delimiter=",")
    loss = [row[-1] for row in loss_table]
    plt.plot(lr, loss, label=f"Weight Decay = {label}", color=color)

plt.xlim([LOW_LR, HIGH_LR])
plt.ylim([6e-4, 1.7e-3])
plt.legend(fontsize=14)
plt.xlabel("Learning Rate", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)