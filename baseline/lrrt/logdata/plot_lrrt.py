#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:52:31 2022

@author: kafkan
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/baseline/lrrt/logdata")
#%%

NUM_SAMPLES = 2867
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIGH_LR = 2
LOW_LR = 1e-5

lr_steps = np.linspace(1e-5, 2, num=NUM_SAMPLES//BATCH_SIZE*NUM_EPOCHS)

lr_no_reg = np.loadtxt("lr_no_reg_b32_e20.csv", skiprows = 1, delimiter=",")
loss_no_reg = np.loadtxt("loss_no_reg_b32_e20.csv", skiprows = 1, delimiter=",")

lr_no_reg = [tup[-1] for tup in lr_no_reg]

loss_steps = [tup[1] for tup in loss_no_reg]
loss_lr = [lr_steps[int(step)] for step in loss_steps ]
loss_no_reg = [tup[-1] for tup in loss_no_reg]

# a = [tup[1] for tup in lr_no_reg]
# b = [tup[1] for tup in loss_no_reg]
# plt.xscale("log")
# plt.xticks([10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 2])

#%% Dropout LRRT
plt.figure(figsize=(5,5))



BASE = "loss_{reg}{val}.csv"

# plt.figure(figsize=(5,5))
names = ["001", "005", "010"]
values = ["0.01", "0.05", "0.10"]
colors = list(sns.color_palette())
for name, value, color in zip(names, values, colors):
    print(BASE.format(reg="dropout",val=name))
    loss_table = np.loadtxt(BASE.format(reg="dropout",val=name), skiprows = 1, delimiter=",")
    loss = [row[-1] for row in loss_table]
    plt.plot(loss_lr, loss, label=f"Dropout = {value}", color=color)

plt.xlim([10e-5, 2])
plt.plot(loss_lr, loss_no_reg, color="k", label="No regularization")
plt.legend()
plt.xlabel("Learning Rate", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=12)

#%% Weight decay LRRT
plt.figure(figsize=(5,5))



BASE = "loss_{reg}{val}.csv"

# 
names = ["001", "0001", "00001", "000001"]
values = ["1e-3", "1e-4", "1e-5",] # "1e-6"]
colors = list(sns.color_palette())
for name, value, color in zip(names, values, colors):
    print(BASE.format(reg="dropout",val=name))
    loss_table = np.loadtxt(BASE.format(reg="wd",val=name), skiprows = 1, delimiter=",")
    loss = [row[-1] for row in loss_table]
    plt.plot(loss_lr, loss, label=f"Weight decay = {value}", color=color)

plt.xlim([10e-5, 2])
plt.plot(loss_lr, loss_no_reg, color="k", label="No regularization")

plt.legend()
plt.xlabel("Learning Rate", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tick_params(labelsize=12)