import sys
sys.path.append("../bin")       # bin for this project
sys.path.append("../../bin")    # general utilities
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# import model
from seqsleepnet_base import SeqSleepNetBase

# import dataloader
from test_utils import load_data, evaluate_test

#%% load data
trainSampler, valSampler, _, valIdxs = load_data(fold=0, return_idxs=True)

#%% Instantiate model
NUM_EPOCHS = 100
NUM_EPOCHS_CYCLE = 10
HIGH_LR = 0.6
LOW_LR = HIGH_LR/10
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.0
L = 20



lr_schedule = torch.optim.lr_scheduler.CyclicLR
base_lr = LOW_LR
steps_half_cycle = trainSampler.sampler.num_samples*NUM_EPOCHS_CYCLE/BATCH_SIZE
schedule_args = {
    "base_lr":base_lr,
    "max_lr":HIGH_LR,
    "step_size_up": steps_half_cycle,
    "mode":"triangular"
    }

base_model = SeqSleepNetBase(L,1,DROPOUT_RATE, base_lr, lr_schedule, schedule_args,WEIGHT_DECAY)    

#%% Train

params = {
    "NUM_EPOCHS":NUM_EPOCHS,
    "NUM_EPOCHS_CYCLE":NUM_EPOCHS_CYCLE,
    "HIGH_LR":HIGH_LR,
    "LOW_LR":LOW_LR,
    "BATCH_SIZE":BATCH_SIZE,
    "WEIGHT_DECAY":WEIGHT_DECAY,
    "DROPOUT_RATE":DROPOUT_RATE,
    "L":L,
}

expName = "base-cyclical"
logger = TensorBoardLogger(
    "tb_logs",
    name=expName
    )
logger.log_hyperparams(params)

lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint = ModelCheckpoint(dirpath=f"checkpoints/{expName}/",
                             monitor="valKappa",
                             mode="max",
                             save_top_k=1,
                             )

trainer = pl.Trainer(max_epochs=NUM_EPOCHS,deterministic=True,gpus=1,
                     callbacks=[lr_monitor, checkpoint],
                     logger=logger,
                     benchmark=True, #speeds up training if batch size is constant
                     enable_progress_bar=True
                     )

trainer.fit(base_model,trainSampler, valSampler)


logger.finalize("success")

#%% 

best_base = SeqSleepNetBase.load_from_checkpoint(checkpoint.best_model_path)
trainer.validate(best_base, valSampler)
