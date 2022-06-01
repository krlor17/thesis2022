"""
    Perform 10-fold cross validation for unfrozen base model w. constant placebo encoder
"""

import sys
sys.path.append("../bin")
sys.path.append("../../bin")
sys.path.append("../../cross_validation/bin")
from sequence_rec_extension import SequenceExtension
from cnn_encoder import CNNEncoder

import torch
import pytorch_lightning as pl

from cross_validation import cross_validate




NUM_EPOCHS = 100
NUM_EPOCHS_HALF_CYCLE = 10   # half-cycle
HIGH_LR = 1.2  # 
LOW_LR = HIGH_LR/10
BATCH_SIZE = 32
WEIGHT_DECAY = 0 # 0, 1e-4
L = 20

DROPOUT_RATE_BASE = 0.01
DROPOUT_RATE_ENCODER = 0.10
WEIGHT_DECAY = 1e-4 #0, 1e-4 ,1e-3
LATENT_DIM  = 16



def init_func(sampler):
    """ Initialize model w. correct encoder """
    encoder = CNNEncoder(LATENT_DIM, cal_shape=(129,188), dropout_rate=DROPOUT_RATE_ENCODER)
    lr_schedule = torch.optim.lr_scheduler.CyclicLR
    base_lr = LOW_LR
    steps_half_cycle = sampler.sampler.num_samples*NUM_EPOCHS_HALF_CYCLE/BATCH_SIZE
    schedule_args = {
        "base_lr":base_lr,
        "max_lr":HIGH_LR,
        "step_size_up": steps_half_cycle,
        "mode":"triangular"
        }
    
    return SequenceExtension(encoder, base_lr, lr_schedule, schedule_args, weight_decay=WEIGHT_DECAY, dropout_rate=DROPOUT_RATE_BASE)


model_class = SequenceExtension
exp_name = "sequence-extension"

SAVE_DIR = f"/home/au709601/NOBACKUP/SeqSleepNetExtension/extending_recurrent/cross_validate/rundata/{exp_name}"

cross_validate(init_function=init_func, model_class=model_class,  epochs=NUM_EPOCHS, exp_name=exp_name, savedir=SAVE_DIR)

