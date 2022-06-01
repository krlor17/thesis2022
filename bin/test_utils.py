#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import torch
from extended_loader import ExtSleepContainer, ExtendedDataset, custom_collate_fn
from sklearn.metrics import cohen_kappa_score

#%% auxilliary

def load_from_disk():
    found = False

    #prime:
    tempMat='/home/au709601/NOBACKUP/data/'
    if  not found and os.path.exists(tempMat):
        matDir=tempMat
        found=True
        print(matDir)
        
    #home:
    tempMat='/media/kafkan/storage1/speciale/data/pickle/'
    if  not found and os.path.exists(tempMat):
        matDir=tempMat
        found=True    
        print(matDir)
    
    
    assert(found)
    
    loadedData=ExtSleepContainer.fromDirectory(matDir,'eeg_lr',"calibration_dict.pkl")
    print('Data loaded')
    return loadedData

def _data_samplers(valIdxs, trainIdxs, loadedData, batch_size, nw, L=20):
        # get training and validation data points
        trainX, trainCal, trainCalMap, trainY = loadedData.returnBySubject(trainIdxs)
        valX, valCal, valCalMap, valY = loadedData.returnBySubject(valIdxs)
        
        # reshape labels
        trainY = torch.tensor((trainY-1).flatten()).type(torch.long)
        valY = torch.tensor((valY-1).flatten()).type(torch.long)
        
        # Instantiate databases and samplers
        trainDataset=torch.utils.data.TensorDataset(torch.tensor(trainX),trainY)
        valDataset=torch.utils.data.TensorDataset(torch.tensor(valX), valY)
        trainSampler=torch.utils.data.DataLoader(ExtendedDataset(trainDataset,L, trainCal, trainCalMap),batch_size=batch_size,
                                         shuffle=True,drop_last=True,collate_fn=custom_collate_fn, num_workers=nw)
        valSampler=torch.utils.data.DataLoader(ExtendedDataset(valDataset,L, valCal, valCalMap),batch_size=1,
                                          shuffle=False,drop_last=True,collate_fn=custom_collate_fn, num_workers=1)
        
        return trainSampler, valSampler

#%% interface

def evaluate_test(net, trainer, testIdxs, loadedData, logger, L=20):
    testX, testCal, testCalMap, testLabels = loadedData.returnBySubject(testIdxs)
    testCal = [c.type(torch.float32) for c in testCal]
    ensembleTesting = net.custom_ensemble_test(testX, testCal, testCalMap, testLabels,trainer)
    
    a,b=torch.max(ensembleTesting['ensemble_pred'].cpu().clone().detach(),1)
    kappa=cohen_kappa_score(torch.unsqueeze(b+1,1),testLabels.T)
    
    rolledKappas=np.zeros(L)
    for iRoll in range(L):
        a,b=torch.max(torch.tensor(ensembleTesting['rolled_probs'][iRoll,:,:]),1)
        rolledKappas[iRoll]=cohen_kappa_score(torch.unsqueeze(b+1,1),testLabels.T)
       
    print('Consensus:',testIdxs,kappa)
    logger.log_metrics('subjectKappa',kappa)
    logger.log_metrics('meanRolledKappa',np.mean(rolledKappas))
    
    return kappa

def load_data(fold, batch_size=32, num_workers=8, seed=11, return_idxs=False):
    rng = np.random.RandomState(seed)
    n_subjects = 20
    subj_idxs = np.arange(1, n_subjects+1)
    rng.shuffle(subj_idxs)
    val_fold_list = subj_idxs.reshape(10,2)
    loadedData = load_from_disk()
    valIdxs = val_fold_list[fold]
    trainIdxs = np.delete(val_fold_list, fold).flatten()
    trainSampler, valSampler = _data_samplers(valIdxs, trainIdxs, loadedData, batch_size, num_workers)
    if return_idxs:
        return trainSampler, valSampler, trainIdxs, valIdxs
    return trainSampler, valSampler

def load_cal_spectrograms():
    specs = []
    subjects= []
    
    with open("/media/kafkan/storage1/speciale/data/pickle/calibration_dict.pkl","rb") as file:
        sleep_dict = pickle.load(file)
    
    for subj in range(1,20+1):
        for night in range(1,4+1):
            if (subj, night) in sleep_dict.keys():
                cal = sleep_dict[(subj,night)]
                subjects += [subj]
                specs.append(torch.tensor(cal))
    subjects = np.array(subjects)
    return specs, subjects
