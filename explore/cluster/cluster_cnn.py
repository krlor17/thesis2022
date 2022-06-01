import os
os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/explore/cluster")
import sys
sys.path.append("../bin")
sys.path.append("../../bin")
from collections import Counter
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from exploratory_extension import ExploratoryExtension
from seqsleepnet_base import SeqSleepNetBase

from sklearn.metrics import confusion_matrix
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.decomposition import PCA

# import dataloader
from extended_loader import ExtSleepContainer, ExtendedDataset, custom_collate_fn
from test_utils import load_data, load_cal_spectrograms, load_from_disk

from cluster_utils import pair_conf_metrics, get_sampler, conf_matrix, plot_conf_matrix, hopkins_statistic, adjusted_pair_recall

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
pl.seed_everything(97)

trainSampler, valSampler, trainIdxs, valIdxs = load_data(fold=0, return_idxs=True)
base = SeqSleepNetBase.load_from_checkpoint("../../bin/base.ckpt")
base.freeze()
base.eval()

spectrograms, subjects = load_cal_spectrograms()

plt.rc('font', size=16)

#%% 
MODEL_PATH = "../runs/cnn/checkpoints/cnn-unfrozen/epoch=60-valKappa=0.8629.ckpt"
# MODEL_PATH = "../../cross_validation/cnn/rundata/CNN-fresh/CNN-fresh-fold-2-bestvalKappa=0.7979130148887634.ckpt"

model = ExploratoryExtension.load_from_checkpoint(MODEL_PATH)
model.eval()
model.to(cuda)
# trainer.validate(model, valSampler)
encodings = []

for c in spectrograms:
    c = c.to(cuda).float()
    c = c.reshape((1,)+c.shape)
    encodings.append(np.unique(model.encoder(c).cpu().detach()))

eps = 0.155
# eps = 0.06

cluster = OPTICS(min_samples=4, eps=eps, cluster_method="dbscan")
cluster.fit(encodings)
clustering = cluster_optics_dbscan(reachability=cluster.reachability_,
                                   core_distances=cluster.core_distances_,
                                   ordering=cluster.ordering_,
                                   eps=eps,
                                    )
np.save("labels_cnn", cluster.labels_)
plt.figure(figsize=(6,5))
plt.plot(cluster.reachability_[cluster.ordering_])
plt.hlines(eps, 0, 80, linestyles='dashed', color='red')
plt.xlabel("Cluster-order of nights", fontsize=16)
plt.ylabel("Reachability distance", fontsize=16)

pca = PCA(n_components=3)
projections = pca.fit_transform(encodings)
print("Percentage of variance explained by PCA:",sum(pca.explained_variance_ratio_))
px, py = projections[:,0], projections[:,1]
pz = projections[:,2]

fig, axs = plt.subplots(1,3, figsize=(15,4), sharey=False)

num_clusters = len(np.unique(cluster.labels_))
cmap = ["gray"] + sns.color_palette("tab10")
cmap = cmap[:num_clusters]

plt.rc('font', size=16) 

axs[0].set_xlabel("PC 1"), axs[0].set_ylabel("PC 2")
axs[1].set_xlabel("PC 1"), axs[1].set_ylabel("PC 3")
axs[2].set_xlabel("PC 2"), axs[2].set_ylabel("PC 3")
sns.scatterplot(px,py, hue=cluster.labels_, palette=cmap, ax=axs[0], legend=False)
sns.scatterplot(px,pz, hue=cluster.labels_, palette=cmap, ax=axs[1], legend=False)
sns.scatterplot(py,pz, hue=cluster.labels_, palette=cmap, ax=axs[2], legend=False)
plt.tight_layout()

valmember = (subjects == valIdxs[0]) | (subjects == valIdxs[1])

#%% pair counting matrix
plt.figure(figsize=(5,5))
pair = pair_confusion_matrix(subjects, cluster.labels_ )
ticks = ["Different","Same"]
sns.heatmap(pair, cmap="Blues", 
            annot=True, 
            fmt="d",
            cbar=False,
            xticklabels=ticks,
            yticklabels=ticks)
plt.xlabel("Cluster", fontsize=16)
plt.ylabel("Subject", fontsize=16)


prec, recall, prec_no_noise, recall_no_noise = pair_conf_metrics(subjects, cluster.labels_)

print("Pair Counting Precision:", prec)
print("Pair Counting Recall:", recall)

print("Pair Counting Precision excl. noise:", prec_no_noise)
print("Pair Counting Recall excl. noise:", recall_no_noise)

print("Adjusted recall", adjusted_pair_recall(subjects, cluster.labels_))

#%% Evaluate performance in different clusters

pl.seed_everything(97)
H = hopkins_statistic(encodings)
print("Hopkins statistic:", H)
# indices = np.arange(78)
# C1 = indicies[]
count_labels = Counter(cluster.labels_)
print("Label dist. for train and val:",count_labels)
count_train_labels = Counter(cluster.labels_[valmember == False])
print("Label dist. for train:",count_train_labels)

nights = []
for s in range(1,20+1):
    for n in range(1,4+1):
        if not( ( s==8 and n==3) or (s==11 and n==1) ):
            nights.append([s,n])
nights = np.array(nights)
nights = np.arange(0,78)

Cn1_nights = nights[(valmember == False)&(cluster.labels_ == -1)]
C0_nights = nights[(valmember == False)&(cluster.labels_ == 0)]
C1_nights = nights[(valmember == False)&(cluster.labels_ == 1)]
C2_nights = nights[(valmember == False)&(cluster.labels_ == 2)]
Cn1_idxs = [Cn1_nights]
C0_idxs = [C0_nights]
C1_idxs = [C1_nights]
C2_idxs = [C2_nights]
loadedData = load_from_disk()

L=20

samplern1 = get_sampler(Cn1_idxs, loadedData)
sampler0 = get_sampler(C0_idxs, loadedData)
sampler1 = get_sampler(C1_idxs, loadedData)
sampler2 = get_sampler(C2_idxs, loadedData)
trainer = pl.Trainer(gpus=1)

res0 = trainer.validate(model, sampler0)
res1 = trainer.validate(model, sampler1)
res2 = trainer.validate(model, sampler2)


conf0 = conf_matrix(model, sampler0)
conf1 = conf_matrix(model, sampler1)
conf2 = conf_matrix(model, sampler2)
conf_train = conf_matrix(model, trainSampler)
# to do: get specific nights for each cluster and evaluate 

plot_conf_matrix(conf0)
plot_conf_matrix(conf1)
plot_conf_matrix(conf2)

plot_conf_matrix(conf0, conf_train)
plot_conf_matrix(conf1, conf_train)
plot_conf_matrix(conf2, conf_train)

prec, recall, prec_no_noise, recall_no_noise = pair_conf_metrics(subjects, cluster.labels_)

print("Pair Counting Precision:", prec)
print("Pair Counting Recall:", recall)

print("Pair Counting Precision excl. noise:", prec)
print("Pair Counting Recall excl. noise:", recall)

#%%
# from scipy.stats import multinomial
# NUM_CLUSTERS = 2
# PROBS = [0.5, 0.5]
PROBS = [1/3]*3
PROBS = [0.8, 0.1,0.1]
PROBS =[n/78 for n in Counter(cluster.labels_).values()]

N = 500
conf = np.zeros((2,2))
for _ in range(N):
    rng = np.random.default_rng()
    sample = rng.multinomial( pvals=PROBS, n=78)
    
    rand = []
    for i, n in enumerate(sample):
        rand += [i]*n
    rng.shuffle(rand)
    conf[:,:] +=   pair_confusion_matrix(subjects, rand )/N

conf = np.round(conf).astype(int)

plt.figure(figsize=(5,5))
ticks = ["Same", "Different"]
sns.heatmap(conf, cmap="Greens", 
            annot=True, 
            fmt="d",
            cbar=False,
            xticklabels=ticks,
            yticklabels=ticks)
plt.xlabel("Cluster", fontsize=16)
plt.ylabel("Subject", fontsize=16)


prec, recall, prec_no_noise, recall_no_noise = pair_conf_metrics(subjects, cluster.labels_)

print("Pair Counting Precision:", prec)
print("Pair Counting Recall:", recall)

print("Pair Counting Precision excl. noise:", prec_no_noise)
print("Pair Counting Recall excl. noise:", recall_no_noise)
