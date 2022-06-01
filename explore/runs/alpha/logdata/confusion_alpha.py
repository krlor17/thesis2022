import os
os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/explore/runs/alpha/logdata")
import sys
sys.path.append("../../../bin")
sys.path.append("../../../../bin")
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from exploratory_extension import ExploratoryExtension
from seqsleepnet_base import SeqSleepNetBase
from alpha_encoder import SimpleAlphaEncoder

from sklearn.metrics import confusion_matrix

# import dataloader
from test_utils import load_data

#%% Set up data and GPU

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

#%%
def plot_conf_matrix(model, valSampler, compare=None):
    model.to(cuda)
    true_labels = np.array([])
    pred_labels = np.array([])
    for x, c, y in tqdm(valSampler):
        x=x.to(cuda)
        c = c.float()
        c=c.to(cuda)
        y_pred = model(x, c)
        _, y_pred = torch.max(y_pred, dim=1)
        y_pred = y_pred.cpu().detach().numpy()
        y = y.detach().numpy()
        true_labels = np.concatenate((true_labels, y))
        pred_labels = np.concatenate((pred_labels, y_pred))
    ticks = ["W","R","N1","N2","N3"]
    conf= confusion_matrix(true_labels, pred_labels)
    plt.figure()
    if type(compare) != None:
        sns.heatmap(conf-compare, annot=True, fmt="d", cbar=False,
                    xticklabels=ticks,
                    yticklabels=ticks,
                    annot_kws={"fontsize":14},
                    cmap=sns.diverging_palette(240,10, n=5),
                    center=0
                    )
    else:
        sns.heatmap(conf, annot=True, fmt="d", cbar=False,
                    xticklabels=ticks,
                    yticklabels=ticks,
                    annot_kws={"fontsize":14},
                    cmap="Blues"
                    )
    plt.xlabel("Predicted labels", fontsize=16)
    plt.ylabel("True labels", fontsize=16)

def make_confusion_matrix(model, valSampler):
    model.to(cuda)
    true_labels = np.array([])
    pred_labels = np.array([])
    for x, c, y in tqdm(valSampler):
        x=x.to(cuda)
        c = c.float()
        c=c.to(cuda)
        y_pred = model(x, c)
        _, y_pred = torch.max(y_pred, dim=1)
        y_pred = y_pred.cpu().detach().numpy()
        y = y.detach().numpy()
        true_labels = np.concatenate((true_labels, y))
        pred_labels = np.concatenate((pred_labels, y_pred))
    ticks = ["W","R","N1","N2","N3"]
    conf= confusion_matrix(true_labels, pred_labels)
    return conf

def conf_matrix_metrics(conf, verbose=False):
    metrics = []
    for i, label in enumerate(ticks):
        TP = conf[i,i]
        FP = sum(conf[:,i]) - TP
        FN = sum(conf[i,:]) - TP
        prec = TP/(TP+FP)
        recall = TP/(TP+FN)
        prop = sum(conf[i,:])/np.sum(conf)
        if verbose:
            print(label,f"\t precision:{prec:.4f}\t recall:{recall:.4f}\t")
        metrics.append([label, prec, recall, prop])
    table = pd.DataFrame(metrics, columns=["label","precision", "recall", "proportion"])
    return table


#%%
pl.seed_everything(97)

trainSampler, valSampler, trainIdxs, valIdxs = load_data(fold=0, return_idxs=True)
base = SeqSleepNetBase.load_from_checkpoint("../../../bin/base.ckpt")
base.freeze()
base.eval()
trainer = pl.Trainer(gpus=1)
#%% Base

val_dict = trainer.validate(base, valSampler)
base.to(cuda)
true_labels = np.array([])
pred_labels = np.array([])
for x, c, y in tqdm(valSampler):
    x=x.to(cuda)
    y_pred,_ = base(x)
    _, y_pred = torch.max(y_pred, dim=1)
    y_pred = y_pred.cpu().detach().numpy()
    y = y.detach().numpy()
    true_labels = np.concatenate((true_labels, y))
    pred_labels = np.concatenate((pred_labels, y_pred))
ticks = ["W","R","N1","N2","N3"]
base_conf= confusion_matrix(true_labels, pred_labels)
plt.figure()
sns.heatmap(base_conf, annot=True, fmt="d", cbar=False,
            xticklabels=ticks,
            yticklabels=ticks,
            annot_kws={"fontsize":13},
            cmap="Blues"
            )
plt.xlabel("Predicted labels", fontsize=16)
plt.ylabel("True labels", fontsize=16)

trainer.validate(base, valSampler)
print("base model accuracy:",sum(base_conf[np.eye(5)==1])/np.sum(base_conf))
print(val_dict[0])

for i, label in enumerate(ticks):
    TP = base_conf[i,i]
    FP = sum(base_conf[:,i]) - TP
    FN = sum(base_conf[i,:]) - TP
    prec = TP/(TP+FP)
    recall = TP/(TP+FN)
    prop = sum(base_conf[:,i])/np.sum(base_conf)
    print(label,f"\t precision:{prec:.4f}\t recall:{recall:.4f}\t proportion:{prop:.4f}")


#%% Simple Alpha

MODEL_PATH = "../checkpoints/simple-alpha/epoch=87-valKappa=0.8539.ckpt"
MODEL_PATH = "../checkpoints/simple-alpha-unfrozen/epoch=99-valKappa=0.8694.ckpt"
#
model = ExploratoryExtension.load_from_checkpoint(MODEL_PATH)
model.eval()
trainer.validate(model, valSampler)
conf = make_confusion_matrix(model, valSampler)
plot_conf_matrix(model, valSampler, compare=base_conf)
table = conf_matrix_metrics(conf)
print(table)

#%% Alpha

MODEL_PATH = "../checkpoints/alpha/epoch=14-valKappa=0.8533.ckpt"
MODEL_PATH = "../checkpoints/alpha-unfrozen/epoch=98-valKappa=0.8672.ckpt"


model = ExploratoryExtension.load_from_checkpoint(MODEL_PATH)
model.eval()
trainer.validate(model, valSampler)
conf = make_confusion_matrix(model, valSampler)
plot_conf_matrix(model, valSampler, compare=base_conf)
table = conf_matrix_metrics(conf)
print(table)
