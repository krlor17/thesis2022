import os
os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/explore/cluster")
import sys
sys.path.append("../bin")
sys.path.append("../../bin")
from collections import Counter
import random
import math

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.decomposition import PCA

# import dataloader
from extended_loader import ExtSleepContainer, ExtendedDataset, custom_collate_fn
from test_utils import load_data, load_cal_spectrograms, load_from_disk


def conf_matrix(model, sampler):
    cuda = torch.cuda.current_device()
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
    
def get_sampler(idxs, loadedData, L=20):
    X, cal, calMap, Y = loadedData.returnRecords(idxs)
    Y = torch.tensor((Y-1).flatten()).type(torch.long)
    dataset=torch.utils.data.TensorDataset(torch.tensor(X),Y)
    sampler = torch.utils.data.DataLoader(ExtendedDataset(dataset,L,cal, calMap),batch_size=32,
                                      shuffle=False,drop_last=True,collate_fn=custom_collate_fn, num_workers=8)
    return sampler

def pair_conf_metrics(subjects, labels):
    pair_conf = pair_confusion_matrix(subjects, labels)
    tn = pair_conf[0,0] # nights from diff subjects in diff cluster
    fn = pair_conf[1,0] # nights from same subjects in diff cluster
    tp = pair_conf[1,1] # nights from same subjects in same cluster
    fp = pair_conf[0,1] # nights from diff subjects in same cluster
    precision = tp/(tp + fp) # 
    recall = tp/(tp + fn)    # how many nights from the same subject are in the same cluster?
    
    noise_mask = labels != -1
    no_noise_subj = subjects[noise_mask]
    no_noise_labels = labels[noise_mask]
    noise_conf = pair_confusion_matrix(no_noise_subj, no_noise_labels)
    tn = noise_conf[0,0] # nights from diff subjects in diff cluster
    fn = noise_conf[1,0] # nights from same subjects in diff cluster
    tp = noise_conf[1,1] # nights from same subjects in same cluster
    fp = noise_conf[0,1] # nights from diff subjects in same cluster
    
    prec_noise = tp/(tp + fp)
    recall_noise = tp/(tp + fn)
    
    return precision, recall, prec_noise, recall_noise

def hopkins_statistic(encodings):
    """
    Compute a hopkins statistic cf. 
    Banerjee and Dave 2004 
    Parameters
    ----------
    encodings : list
        List of points.

    Returns
    -------
    H : float
        Hopkins statistic. H=0 grid, H=0.5 random, H ~ 1.0 very strongly clustered

    """
    dim = len(encodings[0])
    n = len(encodings)
    num_samples = math.ceil(0.1*n)
    neighbors = NearestNeighbors(n_neighbors=1).fit(encodings)
    
    sample = random.sample(encodings, num_samples)
    comp_min = np.amin(encodings,axis=0)    
    comp_max = np.amax(encodings,axis=0) 
    
    u = []
    w = []
    for point in sample:
        point = point.reshape(1,-1)
        w_d, _ = neighbors.kneighbors(point, n_neighbors=2, return_distance=True)
        w.append(w_d[0][1])
        rand_point = np.random.uniform(comp_min, comp_max, size=dim).reshape(1,-1)
        u_d, _ = neighbors.kneighbors(rand_point, n_neighbors=2, return_distance=True)
        u.append(u_d[0][1])
    
    H = sum(u) / ( sum(u) + sum(w) )
    return H

def greedy_max_recall(labels):
    """ 
        Greedily estimated max. recall possible for pair counting matrix
        with the given cluster sizes. (Algorithm 2 in thesis)
        Assumes 18 subjects has 4 nights, 2 subjects 3 nights.
        There must be 78 total labels.
    
    """
    assert len(labels) == 78
    def wrong(k, n):
        return (k-n)*n
    counter = Counter(labels)
    sizes = np.array(list(counter.values()))
    residuals = np.mod(sizes, 4)
    residuals.sort()
    residuals = residuals[::-1] # sort descending order
    largest = residuals[0]
    second = residuals[1]
    wrong_pairs_3_nights =  wrong(3,largest) + wrong(3, second)
    num_wrong = wrong_pairs_3_nights + sum(wrong(4,s) for s in residuals[2:])
    num_wrong /= 2 # num unordered pairs
    TOTAL_PAIRS = 18*4*(4-1) + 2*3*(3-1) # 228
    return 1 - num_wrong/TOTAL_PAIRS

def adjusted_pair_recall(subjects, labels):
    pair_conf = pair_confusion_matrix(subjects, labels)
    fn = pair_conf[1,0] # nights from same subjects in diff cluster
    tp = pair_conf[1,1] # nights from same subjects in same cluster
    recall = tp/(tp+fn)
    avg_recall = 0
    N=1000
    shuffled = labels.copy()
    for _ in range(N):
        np.random.shuffle(shuffled)
        conf = pair_confusion_matrix(subjects, shuffled)
        avg_recall += 1/N*conf[1,1]/(conf[1,1]+conf[1,0])

    approx_max_recall = greedy_max_recall(labels)
    print("recall",recall)
    print("avg. recall", avg_recall)
    print("max. recall", approx_max_recall)
    adjusted = (recall - avg_recall)/(approx_max_recall - avg_recall)
    return adjusted