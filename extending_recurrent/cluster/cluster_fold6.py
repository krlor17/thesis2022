import os
os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/extending_recurrent/cluster")
import sys
sys.path.append("/media/kafkan/storage1/SeqSleepNetExtension/extending_recurrent//bin")
sys.path.append("../../bin")
from collections import Counter
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sequence_rec_extension import SequenceExtension

from sklearn.metrics import confusion_matrix
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.decomposition import PCA

# import dataloader
from extended_loader import ExtSleepContainer, ExtendedDataset, custom_collate_fn
from test_utils import load_data, load_cal_spectrograms, load_from_disk

from cluster_utils import pair_conf_metrics, get_sampler, conf_matrix, plot_conf_matrix, hopkins_statistic, adjusted_pair_recall
os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/extending_recurrent/cluster")
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


spectrograms, subjects = load_cal_spectrograms()

plt.rc('font', size=16)

#%% 
MODEL_PATH = "../cross_validate/alpha/rundata/sequence-extension-alpha/sequence-extension-alpha/sequence-extension-alpha-fold-6-bestvalKappa=0.7488229870796204.ckpt"

model = SequenceExtension.load_from_checkpoint(MODEL_PATH)
model.eval()
model.to(cuda)
encodings = []

for c in spectrograms:
    c = c.to(cuda).float()
    c = c.reshape((1,)+c.shape)
    encodings.append(np.unique(model.encoder(c).cpu().detach()))

eps = 0.31

cluster = OPTICS(min_samples=4, eps=eps, cluster_method="dbscan")
cluster.fit(encodings)
clustering = cluster_optics_dbscan(reachability=cluster.reachability_,
                                   core_distances=cluster.core_distances_,
                                   ordering=cluster.ordering_,
                                   eps=eps,
                                    )
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

s3x = px[(3-1)*4:3*4]
s3y = py[(3-1)*4:3*4]
s3z = pz[(3-1)*4:3*4]
s9x = px[(9-1)*4-1:9*4-1] 
s9y = py[(9-1)*4-1:9*4-1] 
s9z = pz[(9-1)*4-1:9*4-1] 
# sns.scatterplot(s3x,s3y, color="red", style=0, s=20, marker="+"  ,ax=axs[0], legend=False)
# sns.scatterplot(s9x,s9y, color="black", ax=axs[0], legend=False)
axs[0].scatter(s3x,s3y, color="red", marker="x", s=120)
axs[0].scatter(s9x,s9y, color="black", marker="1", s=120)
axs[1].scatter(s3x,s3z, color="red", marker="x", s=120)
axs[1].scatter(s9x,s9z, color="black", marker="1", s=120)
axs[2].scatter(s3y,s3z, color="red", marker="x", s=120)
axs[2].scatter(s9y,s9z, color="black", marker="1", s=120)
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

#%%
pl.seed_everything(98)
nights = []
for s in range(1,20+1):
    for n in range(1,4+1):
        if not( ( s==8 and n==3) or (s==11 and n==1) ):
            nights.append([s,n])
nights = np.array(nights)
nights = np.arange(0,78)

s3_nights = nights[2*4:3*4]
s9_nights = nights[8*4-1:9*4-1]

mask3 = cluster.labels_[2*4:3*4] == 1 
mask9 = cluster.labels_[8*4-1:9*4-1] == 1 

nights_in_val = np.concatenate((s3_nights[mask3], s9_nights[mask9]))
nights_out_val = np.concatenate((s3_nights[mask3 == False], s9_nights[mask9 == False]))
nights_in3 = s3_nights[mask3]
nights_out3 = s3_nights[mask3 == False]

nights_in9 = s9_nights[mask9]
nights_out9 = s9_nights[mask9 == False]

s3_idxs = [s3_nights]
s9_idxs = [s9_nights]

in_idxs3 = [nights_in3]
out_idxs3 = [nights_out3]

in_idxs9 = [nights_in9]
out_idxs9 = [nights_out9]


in_idxs_val = [nights_in_val]
out_idxs_val = [nights_out_val]
loadedData = load_from_disk()

L=20

in_sampler_val = get_sampler(in_idxs_val, loadedData)
out_sampler_val = get_sampler(out_idxs_val, loadedData)
# in_sampler9 = get_sampler(in_idxs9, loadedData)
# out_sampler9 = get_sampler(out_idxs9, loadedData)
# s3_sampler = get_sampler(s3_idxs, loadedData)
# s9_sampler = get_sampler(s9_idxs, loadedData)

in_idxs = [nights[cluster.labels_ == 1]]
out_idxs = [nights[cluster.labels_ != 1]]

in_sampler = get_sampler(in_idxs, loadedData)
out_sampler = get_sampler(out_idxs, loadedData)

trainer = pl.Trainer(gpus=1)

# res_in3 = trainer.validate(model, in_sampler3)
# res_out3 = trainer.validate(model, out_sampler3)
# res_in9 = trainer.validate(model, in_sampler9)
# # res_out9 = trainer.validate(model, out_sampler9)
# res3 = trainer.validate(model, in_sampler)
# res9 = trainer.validate(model, out_sampler)

res_in = trainer.validate(model, in_sampler)
res_out = trainer.validate(model, out_sampler)

res_in_val = trainer.validate(model, in_sampler_val)
res_out_val = trainer.validate(model, out_sampler_val)
