"""
    Perform 10-fold cross validation for unfrozen base model w. constant placebo encoder
"""

import sys
sys.path.append("../bin")
sys.path.append("../../bin")

from exploratory_extension import ExploratoryExtension
from seqsleepnet_base import SeqSleepNetBase
from constant_placebo import ConstantPlaceboEncoder

import torch
import pytorch_lightning as pl

from cross_validation import cross_validate

NUM_EPOCHS = 100
NUM_EPOCHS_HALF_CYCLE = 10   # half-cycle
HIGH_LR = 1.0
LOW_LR = HIGH_LR/10
BATCH_SIZE = 32
DROPOUT_RATE = 0.0
WEIGHT_DECAY = 1e-4
L = 20
LATENT_DIM = 16


def init_func(sampler):
    """ Initialize model w. correct encoder """
    base = SeqSleepNetBase(weight_decay=WEIGHT_DECAY, dropOutProb=DROPOUT_RATE)
    encoder = ConstantPlaceboEncoder(LATENT_DIM, cal_shape=(129,188))
    lr_schedule = torch.optim.lr_scheduler.CyclicLR
    base_lr = LOW_LR
    steps_half_cycle = sampler.sampler.num_samples*NUM_EPOCHS_HALF_CYCLE/BATCH_SIZE
    schedule_args = {
        "base_lr":base_lr,
        "max_lr":HIGH_LR,
        "step_size_up": steps_half_cycle,
        "mode":"triangular"
        }
    
    return ExploratoryExtension(base, encoder, base_lr, lr_schedule, schedule_args, weight_decay=WEIGHT_DECAY)


model_class = ExploratoryExtension
exp_name = "constant-unfrozen"

SAVE_DIR = "/home/au709601/NOBACKUP/SeqSleepNetExtension/cross_validation/placebo/rundata"

cross_validate(init_function=init_func, model_class=model_class,  epochs=NUM_EPOCHS, exp_name=exp_name, savedir=SAVE_DIR)
