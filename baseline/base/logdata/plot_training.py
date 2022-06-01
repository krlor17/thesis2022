import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/baseline/base/logdata/")


#%% 

NUM_SAMPLES = 2867
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIGH_LR = 3
LOW_LR = 1e-5


lr_table= np.loadtxt("lr_base.csv", skiprows = 1, delimiter=",")
lr = [row[-1] for row in lr_table]

end = 100

steps = np.arange(end)

train_loss_table = np.loadtxt("train_loss_base.csv", skiprows = 1, delimiter=",")
train_loss = train_loss_table[0:end,2]

train_kappa_table = np.loadtxt("train_kappa_base.csv", skiprows = 1, delimiter=",")
train_kappa = train_kappa_table[0:end,2]

val_loss_table = np.loadtxt("val_loss_base.csv", skiprows = 1, delimiter=",")
val_loss = val_loss_table[0:end,2]

val_kappa_table = np.loadtxt("val_kappa_base.csv", skiprows = 1, delimiter=",")
val_kappa = val_kappa_table[0:end,2]


#%% kappa

plt.figure(figsize=(5,5))
plt.plot(steps, train_kappa, label="training")
plt.plot(steps, val_kappa, label="validation")
# plt.xlim([LOW_LR, HIGH_LR])
plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Cohen's Kappa", fontsize=16)
# plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)

#%% loss

plt.figure(figsize=(5,5))
plt.plot(steps, train_loss, label="training")
# plt.plot(steps, val_loss, label="validation")
# plt.xlim([LOW_LR, HIGH_LR])
# plt.legend(fontsize=14)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=14)

