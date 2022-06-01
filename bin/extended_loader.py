#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:24:26 2022

@author: kafkan
"""
import numpy as np
import h5py
import os
import pickle
from pathlib import Path
import torch
from loadMat5 import loadMatData


class ExtSleepContainer:
    def __init__(self,sleepDict, calibrationDict):
        self.Xlist=sleepDict['Xlist']
        self.labelList=sleepDict['labelList']
        self.subjectName=sleepDict['subjectName']
        self.subjectNight=sleepDict['subjectNight']
        self.n=len(self.Xlist)
        self.normalize()
        self.calDict = calibrationDict

    def __repr__(self):
        return 'Dataset with ' + str(self.n) + ' recordings'
    
    def normalize(self):
        self.n=len(self.Xlist)
        assert self.n == len(self.labelList)
        
        #normalize data (for each frequency):
        allMeans=np.array([np.mean(x,axis=(1,2)) for x in self.Xlist if len(x)>10])
        totMean=np.mean(allMeans,0).reshape((-1,1,1))
        Xlist=[x-totMean for x in self.Xlist]
        
        allStds=np.array([np.std(x,axis=(1,2)) for x in self.Xlist if len(x)>10])
        totStd=np.mean(allStds,0).reshape((-1,1,1))
        self.Xlist=[x/totStd for x in Xlist]
        
    
    @classmethod
    def fromDirectory(cls,directory, deriv, calibration_file):
        sleepDict=loadMatData(directory,deriv)
        with open(directory+calibration_file, "rb") as file:
            calibrationDict = pickle.load(file)
        
        return cls(sleepDict, calibrationDict)

        
    def returnRecords(self,idxs):
        idxs=idxs[0]
        assert np.array(idxs).size

        #ignore empty idxs:
        idxs=[idxs[i] for i in range(len(idxs)) if self.Xlist[idxs[i]].size>1000]
        assert np.array(idxs).size
        
        night_tuples = []
        for i in idxs:
            night_tuples.append((self.subjectName[i],self.subjectNight[i]))

        cal = [torch.tensor(self.calDict[tuple(n_tup)]) for n_tup in night_tuples]
        
        
        # keep track of which calibration sequence each epoch is associated with
        calIdx = 0
        calIndices = np.ones(self.Xlist[idxs[0]].shape[2])*calIdx
        calIdx += 1

        Xout=np.array(self.Xlist[idxs[0]])
        label_out=np.array(self.labelList[idxs[0]])
        

        for i in idxs[1:]:
            Xout=np.concatenate([Xout,self.Xlist[i]],axis=2)
            label_out=np.concatenate([label_out,self.labelList[i]],axis=1)
            
            calIndices = np.concatenate([calIndices, np.ones(self.Xlist[i].shape[2])*calIdx])
            calIndices = calIndices.astype(int)
            calIdx += 1
            
            
        #we want batch x 29 x 129 x 1:
        Xout=Xout.swapaxes(0,2) 
        Xout=np.expand_dims(Xout,3)
        return Xout,cal,calIndices,label_out
    
    def returnBySubject(self,iSs):
        assert np.array(iSs).size
        
        #did the user ask for non-existent subjects:
        recs=np.in1d(iSs,self.subjectName)
        if not all(np.in1d(iSs,self.subjectName)):
            print('Error: requested subject not in data set')
            raise SystemExit(0)    
        
        #find recordings for all subjects:
        recs=np.where(np.in1d(self.subjectName,iSs))
   
        Xout, cal, calIndices, label_out=self.returnRecords(recs)
        return Xout, cal, calIndices, label_out

#%% Quick and dirty hotfix -- exclude 8,3 and 11,1
def loadMatData(matDir,deriv):
    pickleName=os.path.join(matDir,deriv+'_ext_pickled.pkl')
    print('Pickle-name:',pickleName)
    if os.path.exists(pickleName):
        print('Loading pickled data')
        temp=pickle.load(open(pickleName,'rb'))
        Xlist=temp['Xlist']
        labelList=temp['labelList']
        subjectName=temp['subjectName']
        subjectNight=temp['subjectNight']
        
    else:    
        Xlist=[0] #dummy first value
        labelList=[0]
        subjectName=np.empty((0,))
        subjectNight=np.empty((0,))
        counter=0
        #get subject-dirs:
        p = Path(matDir)
        subjectDirs=[x for x in p.iterdir() if x.is_dir()]

        for iS in range(len(subjectDirs)):
            try:
                subjectName_temp=int(str(subjectDirs[iS])[-2:])
            except:
                subjectName_temp=int(iS)
            
            #get night-dirs:
            p =subjectDirs[iS]
            nightDirs=[x for x in p.iterdir() if x.is_dir()]
            for iN in range(len(nightDirs)):
                
                filename = os.path.join(nightDirs[iN], deriv+'.mat')
                temp=h5py.File(filename,'r')
                subjectName=np.append(subjectName,subjectName_temp)
                subjectNight=np.append(subjectNight,iN)
                Xlist+=[np.array(temp['X'])]
                try:
                    labelList+=[np.array(temp['label'])]
                except:
                    #if there are no labels:
                    labelList+=[np.empty((0,0))]
                
                counter+=1
        
        Xlist.pop(0)
        labelList.pop(0)
    
        print('Pickling data')
        pickle.dump({'Xlist':Xlist,'labelList':labelList,'subjectName':subjectName,'subjectNight':subjectNight}, open( pickleName, "wb" ) )
    
    return {'Xlist':Xlist,'labelList':labelList,'subjectName':subjectName,'subjectNight':subjectNight}


#%% Dataset class for extended

class ExtendedDataset(torch.utils.data.Dataset):
    #a wrapper for torch datasets, to make it possible to shuffle sequences
    def __init__(self, eegDataset,sequenceLength, calibrationData, calibrationIdxs):
        
        self.eegData=eegDataset
        self.L=sequenceLength
        self.calData = calibrationData
        self.calMap = calibrationIdxs
        
        #bookkeeping idx's:
        self.epochIdxs=None
        self.calIdxs = None
        self.getCounter=0
        self.reset()
        
    def reset(self):
        #reset bookkeeping idx's
        start=np.random.randint(0,self.L)
        seqRange=range(start,len(self.eegData),self.L)
        seqRange=range(seqRange[0],seqRange[-1])
        self.epochIdxs=np.reshape(seqRange,(-1,self.L))
        self.calIdxs = self.calMap[self.epochIdxs]
        self.getCounter=0        

    # returns number of samples in dataset
    def __len__(self):
        return int(np.floor(len(self.eegData)/self.L)) 
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if type(idx) in (tuple,list):
            print(len(idx))
            idx=idx[0]

        try:
            self.getCounter+=len(idx)
        except:
            #if idx is a scalar, the other one fails
            self.getCounter+=np.array(idx).size
                
        #because __len__ fluctuates, we need to make sure we don't try to access non-existing data:
        idx=idx%(self.epochIdxs.shape[0])
        
        # try:
        sample=self.eegData[np.reshape(self.epochIdxs[idx,:],(-1,))]
        calibration = self.calData[self.calMap[idx]]
        #if all idxs have been passed:
        if self.getCounter >= (self.epochIdxs.shape[0]-1):
            self.reset()

        # X, cal, y
        return sample[0], calibration, sample[1]



#%%

def custom_collate_fn(batch):
    x = torch.cat([item[0] for item in batch])
    c = torch.cat([item[1] for item in batch])
    y = torch.cat([item[2] for item in batch])
    # i = torch.cat([item[2] for item in batch])
    return x, c, y

def test_collate_fn(batch):
    x = torch.cat([item[0] for item in batch])
    c = torch.cat([item[1] for item in batch])
    idx = torch.tensor([item[2] for item in batch])
    return x, c, idx

#%%

class TestDataset(torch.utils.data.Dataset):
  #a wrapper for torch datasets, to make it possible to shuffle sequences
    def __init__(self, eegDataset,sequenceLength, calibrationData, calibrationIdxs):
        
        self.eegData=eegDataset
        self.L=sequenceLength
        self.calData = calibrationData
        self.calMap = calibrationIdxs
        
        #bookkeeping idx's:
        self.epochIdxs=None
        self.calIdxs = None
        self.getCounter=0
        self.reset()
        
    def reset(self):
        #reset bookkeeping idx's
        start=np.random.randint(0,self.L)
        seqRange=range(start,len(self.eegData),self.L)
        seqRange=range(seqRange[0],seqRange[-1])
        self.epochIdxs=np.reshape(seqRange,(-1,self.L))
        self.calIdxs = self.calMap[self.epochIdxs]
        self.getCounter=0        

    # returns number of samples in dataset
    def __len__(self):
        return int(np.floor(len(self.eegData)/self.L)) 
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if type(idx) in (tuple,list):
            print(len(idx))
            idx=idx[0]

        try:
            self.getCounter+=len(idx)
        except:
            #if idx is a scalar, the other one fails
            self.getCounter+=np.array(idx).size
                
        #because __len__ fluctuates, we need to make sure we don't try to access non-existing data:
        sample=self.eegData[np.reshape(self.epochIdxs[idx,:],(-1,))]
        calibration = self.calData[self.calMap[idx]]
        #if all idxs have been passed:
        if self.getCounter >= (self.epochIdxs.shape[0]-1):
            self.reset()

        # X, cal, y
        return sample, calibration, self.epochIdxs[idx,:]


if __name__ == '__main__':
    matDir='/media/kafkan/storage1/speciale/data/pickle/'

    deriv='eeg_lr'
    loadedData=ExtSleepContainer.fromDirectory(matDir,deriv, "calibration_dict.pkl")
    
    trainIdxs=np.arange(2,16)
    valIdxs=np.arange(16,20+1)
    
    trainX, cal, calmap, trainY = loadedData.returnBySubject(trainIdxs)
    trainY = (trainY -1 ).flatten()
    trainTensorData = torch.utils.data.TensorDataset(torch.tensor(trainX),torch.tensor(trainY))
    trainData = ExtendedDataset(trainTensorData, 20, cal, calmap)
