import os
os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/explore/cluster")
import sys
sys.path.append("../../bin")
sys.path.append("../../../bin")
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
from alpha_encoder import SimpleAlphaEncoder

from sklearn.metrics import confusion_matrix
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import pair_confusion_matrix

# import dataloader
from extended_loader import ExtSleepContainer, ExtendedDataset, custom_collate_fn
from test_utils import load_data, load_cal_spectrograms, load_from_disk
from cluster_utils import hopkins_statistic, pair_conf_metrics, adjusted_pair_recall

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


# Subject 8 and subject 11 forgot one night each
subjects = []

for s in range(1,20+1):
    if s not in [8,11]:
        subjects += [s]*4
    else:
        subjects += [s]*3

subjects = np.array(subjects)

spectrograms, subjects = load_cal_spectrograms()

#%% Simple Alpha
encoder = SimpleAlphaEncoder(cal_shape=(188,129), timemarks=[28,48,68,88,108]).to(cuda)
encodings = []
for c in spectrograms:
    c = c.to(cuda).float()
    c = c.reshape((1,)+c.shape)
    encodings.append(np.unique(encoder(c).cpu().detach()))
encodings = np.array(encodings)

eps = 0.24
cluster = OPTICS(min_samples=4, eps=eps, cluster_method="dbscan")
cluster.fit(encodings)

###### reachability plot
plt.figure(figsize=(6,5))
plt.plot(cluster.reachability_[cluster.ordering_])
plt.hlines(eps, 0, 80, linestyles='dashed', color='red')
plt.xlabel("Cluster-order of nights", fontsize=16)
plt.ylabel("Reachability distance", fontsize=16)
plt.gca().yaxis.set_major_formatter('{x:.2f}')
plt.tick_params(labelsize=16)
######
pca = PCA(n_components=3)
projections = pca.fit_transform(encodings)
print("Percentage of variance explained by PCA:",sum(pca.explained_variance_ratio_))
px, py = projections[:,0], projections[:,1]
pz = projections[:,2]
# plt.scatter(px,py)
fig, axs = plt.subplots(1,3, figsize=(15,4), sharey=False)
# axs[0].scatter(px,py)
# axs[1].scatter(px,pz)
# axs[2].scatter(py,pz)

num_clusters = len(np.unique(cluster.labels_))
cmap = ["gray"] + sns.color_palette("tab10")
cmap = cmap[:num_clusters]
axs[0].set_xlabel("PC 1"), axs[0].set_ylabel("PC 2")
axs[1].set_xlabel("PC 1"), axs[1].set_ylabel("PC 3")
axs[2].set_xlabel("PC 2"), axs[2].set_ylabel("PC 3")
sns.scatterplot(x=px,y=py, hue=cluster.labels_, palette=cmap, ax=axs[0], legend=None)
sns.scatterplot(x=px,y=pz, hue=cluster.labels_, palette=cmap, ax=axs[1], legend=None)
sns.scatterplot(x=py,y=pz, hue=cluster.labels_, palette=cmap, ax=axs[2], legend=None)

plt.rc('font', size=16) 
plt.tight_layout()

np.save("labels_simple", cluster.labels_)

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
# fig, axs = plt.subplots(1,3, figsize=(15,4), sharey=False)
# cmap = sns.color_palette("tab20")
# cmap = cmap[:20]
# axs[0].set_xlabel("PC 1"), axs[0].set_ylabel("PC 2")
# axs[1].set_xlabel("PC 1"), axs[1].set_ylabel("PC 3")
# axs[2].set_xlabel("PC 2"), axs[1].set_ylabel("PC 3")
# sns.scatterplot(x=px,y=py, hue=subjects, palette=cmap, ax=axs[0], legend=None)
# sns.scatterplot(x=px,y=pz, hue=subjects, palette=cmap, ax=axs[1], legend=None)
# sns.scatterplot(x=py,y=pz, hue=subjects, palette=cmap, ax=axs[2], legend=None)


# conf = make_confusion_matrix(model, valSampler)
# plot_conf_matrix(model, valSampler)
# plot_conf_matrix(model, valSampler, compare=base_conf)
# table = conf_matrix_metrics(conf)
# print(table)
# plt.figure()
# x = np.sum(encodings[:,[0,1]], axis=1)
# y = np.sum(encodings[:,[2,3]], axis=1)
# sns.scatterplot(x,y, hue=subjects, palette="tab20", legend=None)
#%% 
MODEL_PATH = "../runs/alpha/checkpoints/alpha/epoch=14-valKappa=0.8533.ckpt"
MODEL_PATH = "../runs/alpha/checkpoints/alpha-unfrozen/epoch=98-valKappa=0.8672.ckpt"

model = ExploratoryExtension.load_from_checkpoint(MODEL_PATH)
model.eval()
model.to(cuda)
# trainer.validate(model, valSampler)
encodings = []
# for c in trainSampler.dataset.calData:
#     c = c.to(cuda).float()
#     c = c.reshape((1,)+c.shape)
#     encodings.append(np.unique(model.encoder(c).cpu().detach()))
    
# for c in valSampler.dataset.calData:
#     c = c.to(cuda).float()
#     c = c.reshape((1,)+c.shape)
#     encodings.append(np.unique(model.encoder(c).cpu().detach()))

for c in spectrograms:
    c = c.to(cuda).float()
    c = c.reshape((1,)+c.shape)
    encodings.append(np.unique(model.encoder(c).cpu().detach()))

#eps = 0.08
eps = 0.3
cluster = OPTICS(min_samples=4, eps=eps, cluster_method="dbscan")
cluster.fit(encodings)
clustering = cluster_optics_dbscan(reachability=cluster.reachability_,
                                   core_distances=cluster.core_distances_,
                                   ordering=cluster.ordering_,
                                   eps=eps,
                                    )

np.save("labels_alpha", cluster.labels_)
# reach plot
plt.figure(figsize=(6,5))
plt.plot(cluster.reachability_[cluster.ordering_])
plt.hlines(eps, 0, 80, linestyles='dashed', color='red')
plt.xlabel("Cluster-order of nights", fontsize=16)
plt.ylabel("Reachability distance", fontsize=16)
# plt.ticklabel_format(axis='y', style='sci', scilimits=(2,2))
# plt.ticklabel_format(axis='y', style='plain')
plt.gca().yaxis.set_major_formatter('{x:.2f}')
plt.tick_params(labelsize=16)

pca = PCA(n_components=3)
projections = pca.fit_transform(encodings)
print("Percentage of variance explained by PCA:",sum(pca.explained_variance_ratio_))
px, py = projections[:,0], projections[:,1]
pz = projections[:,2]

fig, axs = plt.subplots(1,3, figsize=(15,4), sharey=False)

num_clusters = len(np.unique(cluster.labels_))
cmap = ["gray"] + sns.color_palette("tab10")
cmap = cmap[:num_clusters]
axs[0].set_xlabel("PC 1"), axs[0].set_ylabel("PC 2")
axs[1].set_xlabel("PC 1"), axs[1].set_ylabel("PC 3")
axs[2].set_xlabel("PC 2"), axs[2].set_ylabel("PC 3")
plt.rc('font', size=16) 

sns.scatterplot(px,py, hue=cluster.labels_, palette=cmap, ax=axs[0], legend=False)
sns.scatterplot(px,pz, hue=cluster.labels_, palette=cmap, ax=axs[1], legend=False)
sns.scatterplot(py,pz, hue=cluster.labels_, palette=cmap, ax=axs[2], legend=False)

plt.tight_layout()
# axs[1].scatter(px,pz)
# axs[2].scatter(py,pz)

# axs[0].scatter(px[-7:],py[-7:], color="red")
# axs[1].scatter(px[-7:],pz[-7:], color="red")
# axs[2].scatter(py[-7:],pz[-7:], color="red")
# fig, axs = plt.subplots(1,3,figsize=(18,5))
# cmap = sns.color_palette("tab20")
# cmap = cmap[:20]
# axs[0].set_xlabel("PC 1"), axs[0].set_ylabel("PC 2")
# axs[1].set_xlabel("PC 1"), axs[1].set_ylabel("PC 3")
# axs[2].set_xlabel("PC 2"), axs[1].set_ylabel("PC 3")


# sns.scatterplot(x=px,y=py, hue=subjects, palette=cmap, ax=axs[0], legend=None)
# sns.scatterplot(x=px,y=pz, hue=subjects, palette=cmap, ax=axs[1], legend=None)
# sns.scatterplot(x=py,y=pz, hue=subjects, palette=cmap, ax=axs[2], legend=None)

# valmember = np.array((subjects == valIdxs[0]) | (subjects == valIdxs[1]), dtype=int)
valmember = (subjects == valIdxs[0]) | (subjects == valIdxs[1])

# plt.figure()
# sns.scatterplot(x=px,y=py, hue=valmember, palette=cmap[:2], ax=axs[0], legend=None)
# sns.scatterplot(x=px,y=pz, hue=valmember, palette=cmap[:2], ax=axs[1], legend=None)
# sns.scatterplot(x=py,y=pz, hue=valmember, palette=cmap[:2], ax=axs[2], legend=None)
# plt.rc('font', size=16) 
# plt.tight_layout()
#%% pair counting matrix
plt.figure(figsize=(5,5))
pair = pair_confusion_matrix(subjects, cluster.labels_ )
ticks = ["Different", "Same"]
sns.heatmap(pair, cmap="Blues", 
            annot=True, 
            fmt="d",
            cbar=False,
            xticklabels=ticks,
            yticklabels=ticks)
plt.xlabel("Cluster", fontsize=16)
plt.ylabel("Subject", fontsize=16)


prec, recall, prec_no_noise, recall_no_noise = pair_conf_metrics(subjects, cluster.labels_)

# PROBS =[n/78 for n in Counter(cluster.labels_).values()]

# N = 500
# conf = np.zeros((2,2))
# for _ in range(N):
#     rng = np.random.default_rng()
#     sample = rng.multinomial( pvals=PROBS, n=78)
    
#     rand = []
#     for i, n in enumerate(sample):
#         rand += [i]*n
#     rng.shuffle(rand)
#     conf[:,:] +=   pair_confusion_matrix(subjects, rand )/N

# conf = np.round(conf).astype(int)

# plt.figure(figsize=(5,5))
# ticks = ["Same", "Different"]
# sns.heatmap(conf, cmap="Reds", 
#             annot=True, 
#             fmt="d",
#             cbar=False,
#             xticklabels=ticks,
#             yticklabels=ticks)
# plt.xlabel("Cluster", fontsize=16)
# plt.ylabel("Subject", fontsize=16)

print("Pair Counting Precision:", prec)
print("Pair Counting Recall:", recall)

print("Pair Counting Precision excl. noise:", prec_no_noise)
print("Pair Counting Recall excl. noise:", recall_no_noise)

print("Adjusted recall", adjusted_pair_recall(subjects, cluster.labels_))

#%% Evaluate performance in different clusters

def conf_matrix(model, sampler):
    model.to(cuda)
    conf = np.zeros((5,5))
    for x,c,y in sampler:
        x,c = x.float().to(cuda), c.float().to(cuda)
        _,ypred = torch.max(model(x,c),1)
        ypred = ypred.cpu().detach().numpy()
        temp = confusion_matrix(y, ypred, labels=[0,1,2,3,4] )
        m = np.max(temp)+1
        conf[0:m,0:m] += temp
    return conf
    
    
def plot_conf_matrix(conf, conf2=None, norm = True ):
    conf = np.array(conf, dtype=int)   
    ticks = ["W","R","N1","N2","N3"]
    plt.figure()
    if np.array(conf2).size > 1:
        conf2= np.array(conf2, dtype=int)
        mat = conf/np.sum(conf) - conf2/np.sum(conf2)
        print(np.sum(mat))
        sns.heatmap(mat, annot=True, fmt=".3f", cbar=False,
                    xticklabels=ticks,
                    yticklabels=ticks,
                    annot_kws={"fontsize":14},
                    cmap=sns.diverging_palette(240,10, n=5),
                    center=0
                    )
    elif norm:
        mat = conf/np.sum(conf)
        sns.heatmap(mat, annot=True, fmt=".3f", cbar=False,
                    xticklabels=ticks,
                    yticklabels=ticks,
                    annot_kws={"fontsize":14},
                    cmap="Blues"
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

pl.seed_everything(97)
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

C1_nights = nights[(valmember == False)&(cluster.labels_ != 0)]
C0_nights = nights[(valmember == False)&(cluster.labels_ == 0)]
C1_idxs = [C1_nights]
C0_idxs = [C0_nights]
loadedData = load_from_disk()

L=20
X1, cal1, calMap1, Y1 = loadedData.returnRecords(C1_idxs)
Y1 = torch.tensor((Y1-1).flatten()).type(torch.long)
dataset1=torch.utils.data.TensorDataset(torch.tensor(X1),Y1)
sampler1 = torch.utils.data.DataLoader(ExtendedDataset(dataset1,L,cal1, calMap1),batch_size=32,
                                  shuffle=False,drop_last=True,collate_fn=custom_collate_fn, num_workers=8)

X0, cal0, calMap0, Y0 = loadedData.returnRecords(C0_idxs)
Y0 = torch.tensor((Y0-1).flatten()).type(torch.long)
dataset0=torch.utils.data.TensorDataset(torch.tensor(X0),Y0)
sampler0 = torch.utils.data.DataLoader(ExtendedDataset(dataset0,L,cal0, calMap0),batch_size=32,
                                  shuffle=False,drop_last=True,collate_fn=custom_collate_fn, num_workers=8)

trainer = pl.Trainer(gpus=1)

res1 = trainer.validate(model, sampler1)
res2 = trainer.validate(model, sampler0)
res3 = trainer.validate(model, trainSampler)

conf1 = conf_matrix(model, sampler1)
conf0 = conf_matrix(model, sampler0)
conf_train = conf_matrix(model, trainSampler)
# to do: get specific nights for each cluster and evaluate 

plot_conf_matrix(conf1)
plot_conf_matrix(conf0)
plot_conf_matrix(conf1, conf0)
plot_conf_matrix(conf1, conf_train)
plot_conf_matrix(conf0, conf_train)
