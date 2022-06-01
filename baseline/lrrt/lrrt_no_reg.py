#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning Rate Range Test for SeqSleepNet

Created on Wed Mar  9 16:38:37 2022

@author: kafkan
"""

#%% Imports
import sys
sys.path.append("../bin")       # bin for this project
sys.path.append("../../bin")    # general utilities
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import scipy.special

# import model
from seqsleepnet_base import SeqSleepNetBase

# import dataloader
from extended_loader import ExtSleepContainer, ExtendedDataset, custom_collate_fn

#%% set up torch on GPU if available

cuda=torch.device('cpu')

if torch.cuda.is_available():
    cuda=torch.device('cuda:0')
    print(torch.version.cuda)
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f/1e6)
else:
    print("no cuda available")

#%% load data

found = False

#prime:
tempMat='../data/'
if  not found and os.path.exists(tempMat):
    matDir=tempMat
    found=True
    print(matDir)
    
#home:
tempMat='/media/kafkan/storage1/speciale/data/pickle/'
if  not found and os.path.exists(tempMat):
    matDir=tempMat
    found=True    
    print(matDir)


assert(found)

#comes pre-normalized:
loadedData=ExtSleepContainer.fromDirectory(matDir,'eeg_lr',"calibration_dict.pkl")
print('Data loaded')



#%% 

# parameters
NUM_EPOCHS = 20
L = 20

LOW_LR = 1e-5
HIGH_LR = 2
# BATCH_SIZE = 16
BATCH_SIZE = 32
SEED = 4
WEIGHT_DECAY = 0.0
DROPOUT_RATE = 0.0

#%% neptune

expName="LRRT-no-reg-step"
params = {'L': L,
      'learning_rate_upper': HIGH_LR,
      'learning_rate_lower': HIGH_LR,
      'seed': SEED,
      'weight_decay': WEIGHT_DECAY,
      'dropOutProb': DROPOUT_RATE,
      'expName': expName
      }
    
logger = TensorBoardLogger(
    "tb_logs",
    name=expName
    )
logger.log_hyperparams(params)
    
#%% Data samplers
pl.seed_everything(SEED)
rng = np.random.RandomState(SEED)

#training, test
subjectIdxs = np.array(range(1,20+1))
rng.shuffle(subjectIdxs)
testIdxs = subjectIdxs[0:2]
trainIdxs = subjectIdxs[2:]

#load data
trainX, trainCal, trainCalMap, trainY = loadedData.returnBySubject(trainIdxs)
testX, testCal, testCalMap, testY = loadedData.returnBySubject(testIdxs)

trainY = torch.tensor((trainY-1).flatten()).type(torch.long)
testY = torch.tensor((testY-1).flatten()).type(torch.long)
trainDataset=torch.utils.data.TensorDataset(torch.tensor(trainX),trainY)
testDataset=torch.utils.data.TensorDataset(torch.tensor(testX),testY)
#dataLoaders:
trainSampler=torch.utils.data.DataLoader(ExtendedDataset(trainDataset,L, trainCal, trainCalMap),batch_size=BATCH_SIZE,
                                         shuffle=True,drop_last=True,collate_fn=custom_collate_fn, num_workers=8)
testSampler=torch.utils.data.DataLoader(ExtendedDataset(testDataset,L, testCal, testCalMap),batch_size=BATCH_SIZE,
                                         shuffle=False,drop_last=True,collate_fn=custom_collate_fn, num_workers=8)



#%% Network and training

print("make a clean net:")

lr_schedule = torch.optim.lr_scheduler.CyclicLR
base_lr = LOW_LR
total_steps_for_all_epochs = trainSampler.sampler.num_samples*NUM_EPOCHS/BATCH_SIZE
schedule_args = {
    "base_lr":base_lr,
    "max_lr":HIGH_LR,
    "step_size_up": total_steps_for_all_epochs, #steadily increasing untill termination
    "mode":"triangular"
    }
net=SeqSleepNetBase(L,1,DROPOUT_RATE, base_lr, lr_schedule, schedule_args,WEIGHT_DECAY)    

#start training:
lr_monitor = LearningRateMonitor(logging_interval='step')


trainer = pl.Trainer(max_epochs=NUM_EPOCHS,deterministic=True,gpus=1,
                     callbacks=[lr_monitor],
                     logger=logger,
                     log_every_n_steps=2,
                     benchmark=True, 
                     enable_progress_bar=True
                     )

trainer.fit(net,trainSampler, testSampler)


logger.finalize("success")
