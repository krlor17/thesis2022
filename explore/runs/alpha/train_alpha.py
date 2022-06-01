import sys
sys.path.append("../../bin")       # bin for this project
sys.path.append("../../../bin")    # general utilities

# import DL framework
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

#import models
from exploratory_extension import ExploratoryExtension
from seqsleepnet_base import SeqSleepNetBase
from alpha_encoder import AlphaEncoder

# import dataloader & utils
from test_utils import load_data

SEED = 97
pl.seed_everything(SEED)
#%% Set up GPU
cuda=torch.device('cpu')

if torch.cuda.is_available():
    cuda=torch.device('cuda:0')
    print(torch.version.cuda)
    print(torch.cuda.current_device())
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

trainSampler, valSampler, _, valIdxs = load_data(fold=0, return_idxs=True)

#%% Instantiate model

NUM_EPOCHS = 100
NUM_EPOCHS_CYCLE = 10   # half-cycle
HIGH_LR = 0.6 
LOW_LR = HIGH_LR/10
LATENT_DIM = 16
L = 20
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
TIMEMARKS = [28, 108] 



lr_schedule = torch.optim.lr_scheduler.CyclicLR
base_lr = LOW_LR
steps_half_cycle = trainSampler.sampler.num_samples*NUM_EPOCHS_CYCLE/BATCH_SIZE
schedule_args = {
    "base_lr":base_lr,
    "max_lr":HIGH_LR,
    "step_size_up": steps_half_cycle,
    "mode":"triangular"
    }

base = SeqSleepNetBase.load_from_checkpoint("../../bin/base.ckpt")
# base.freeze() # freeze weights! Comment to train entire network
encoder = AlphaEncoder(LATENT_DIM, cal_shape=(188,129), timemarks=TIMEMARKS)
model = ExploratoryExtension(base, encoder,
                             base_lr = base_lr,
                             lr_sched=lr_schedule,
                             sched_args=schedule_args,
                             weight_decay=WEIGHT_DECAY, 
                             log_every_step = False)

#%% Logger

expName = "alpha-unfrozen"
params = {
        'L': L,
        'learning_rate_upper': HIGH_LR,
        'learning_rate_lower': LOW_LR,
        'batch_size': BATCH_SIZE,
        'weightDecay': WEIGHT_DECAY,
        'latentDim': LATENT_DIM,
        'expName': expName,
        'seed':SEED
        }

logger = TensorBoardLogger(
    "tb_logs",
    name=expName
    )
logger.log_hyperparams(params)

#%% Training

checkpoint = ModelCheckpoint(dirpath=f"checkpoints/{expName}/",
                             filename="{epoch}-{valKappa:.4f}",
                             monitor="valKappa",
                             mode="max",
                             save_top_k=1,
                             )

trainer = pl.Trainer(max_epochs=NUM_EPOCHS,deterministic=True,gpus=1,
                     callbacks=[checkpoint],
                     logger=logger,
                     benchmark=True,
                     enable_progress_bar=True
                     )

trainer.fit(model, trainSampler, valSampler)
logger.finalize("success")
