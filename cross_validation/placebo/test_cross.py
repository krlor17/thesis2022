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


NUM_EPOCHS = 1
NUM_EPOCHS_HALF_CYCLE = 10   # half-cycle
HIGH_LR = 0.4 
LOW_LR = HIGH_LR/10
BATCH_SIZE = 32
DROPOUT_RATE = 0.0
WEIGHT_DECAY = 1e-4
L = 20
LATENT_DIM = 16

lr_schedule = torch.optim.lr_scheduler.CyclicLR
base_lr = LOW_LR
schedule_args = {
    "base_lr":base_lr,
    "max_lr":HIGH_LR,
    "mode":"triangular"
    }

base = SeqSleepNetBase.load_from_checkpoint("../../bin/base.ckpt")
# base.freeze() # freeze weights!
encoder =  ConstantPlaceboEncoder(latent_dim=LATENT_DIM, cal_shape=(129,188))
model_class = ExploratoryExtension
model_args = {
    "base":base,
    "encoder": encoder,
    "base_lr": LOW_LR,
    "weight_decay": WEIGHT_DECAY,
    "log_every_step": False,
}
exp_name = "test"

SAVE_DIR = "/home/au709601/NOBACKUP/SeqSleepNetExtension/cross_validation/placebo/rundata"
SAVE_DIR = "/media/kafkan/storage1/SeqSleepNetExtension/cross_validation/test"

cross_validate(model_class, model_args, lr_schedule, schedule_args, epochs=NUM_EPOCHS, exp_name=exp_name, savedir=SAVE_DIR)

