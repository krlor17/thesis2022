#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning Rate Range Test for Exploratory extension with Tied RNN scheme

@author: krlor17
"""

#%% Imports
import sys
sys.path.append("../../bin")       # bin for this project
sys.path.append("../../../bin")    # general utilities

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

# import model
from sequence_rec_extension import SequenceExtension
from alpha_encoder import AlphaEncoder

# import dataloader
from test_utils import load_data
SEED = 97
pl.seed_everything(SEED)
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


# parameters
NUM_EPOCHS = 20
L = 20

LOW_LR = 1e-5
HIGH_LR = 2
BATCH_SIZE = 32

DROPOUT_RATE_BASE = 0.0
WEIGHT_DECAY = 1e-3
LATENT_DIM  = 16
TIMEMARKS = [28, 108] 



#%% logger

expName="lrrt-sequence-alpha"
params = {'L': L,
      'learning_rate_upper': HIGH_LR,
      'learning_rate_lower': LOW_LR,
      'seed': SEED,
      'dropOutBase': DROPOUT_RATE_BASE,    
      'weightDecay': WEIGHT_DECAY,
      'latentDim': LATENT_DIM,
      'expName': expName,
      }
    
logger = TensorBoardLogger(
    "tb_logs",
    name=expName
    )
logger.log_hyperparams(params)
    
#%% Data samplers
trainSampler, testSampler = load_data(fold=0)


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

encoder = AlphaEncoder(latent_dim=LATENT_DIM, cal_shape=(129,188), timemarks=TIMEMARKS)
model = SequenceExtension(encoder,
                          base_lr = LOW_LR,
                          lr_sched=lr_schedule,
                          sched_args=schedule_args,
                          weight_decay=WEIGHT_DECAY, 
                          dropout_rate = DROPOUT_RATE_BASE,
                          log_every_step = False)

#start training:
lr_monitor = LearningRateMonitor(logging_interval='step')


trainer = pl.Trainer(max_epochs=NUM_EPOCHS,deterministic=True,gpus=1,
                     logger=logger,
                     benchmark=True,
                     enable_progress_bar=True
                     )

trainer.fit(model, trainSampler, testSampler)


logger.finalize("success")
