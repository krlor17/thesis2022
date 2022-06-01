#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: kafkan
"""

import sys
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from filterbank_shape import FilterbankShape 
from sklearn.metrics import cohen_kappa_score
import scipy


class SeqSleepNetBase(pl.LightningModule):
    def __init__(self,L=20,nChan=1,dropOutProb=0.01,base_lr=1e-3,lr_sched=None, sched_args=None,weight_decay=1e-3):
        super().__init__()
         
        #save input:
        self.save_hyperparameters()

        #settings:
        self.L=L #sequence length
        self.nChan=nChan
        self.base_lr=base_lr
        self.lr_sched = lr_sched
        self.sched_args = sched_args
        self.weight_decay=weight_decay
        
        self.nHidden=64
        self.nFilter=32
        self.attentionSize=64
        self.dropOutProb=dropOutProb
        self.timeBins=29

        #---------------------------filterbank:--------------------------------
        filtershape = FilterbankShape()
        
        #triangular filterbank shape
        shape=torch.tensor(filtershape.lin_tri_filter_shape(nfilt=self.nFilter,
                                                            nfft=256,
                                                            samplerate=100,
                                                            lowfreq=0,
                                                            highfreq=50),dtype=torch.float)
        
        self.Wbl = nn.Parameter(shape,requires_grad=False)
        #filter weights:
        self.Weeg = nn.Parameter(torch.randn(self.nFilter,self.nChan))
        #----------------------------------------------------------------------

        self.epochrnn = nn.GRU(self.nFilter,self.nHidden,1,bidirectional=True,batch_first=True)

        #attention-layer:       
        self.attweight_w = nn.Parameter(torch.randn(2*self.nHidden, self.attentionSize))
        self.attweight_b = nn.Parameter(torch.randn(self.attentionSize))
        self.attweight_u = nn.Parameter(torch.randn(self.attentionSize))
        
        #epoch sequence block:
        self.seqDropout = torch.nn.Dropout(self.dropOutProb, inplace=False)
        self.seqRnn = nn.GRU(self.nHidden*2, self.nHidden, 1, bidirectional=True, batch_first=True)
        
        #output:
        self.fc = nn.Linear(2*self.nHidden,5)

    def forward(self, x):

        assert (x.shape[0]/self.L).is_integer() # we need to pass a multiple of L epochs
        assert (x.shape[1]==self.timeBins)
        assert (x.shape[2]==129)
        assert (x.shape[3]==self.nChan)
        
        x = x.permute([0,3,1,2])  # switch ordering to (Batch, Time, Freq, Channel)

        #filtering:
        Wfb = torch.multiply(torch.sigmoid(self.Weeg[:,0]),self.Wbl)
        x   = torch.matmul(x, Wfb) # filtering
        x   = torch.reshape(x, [-1, self.timeBins, self.nFilter]) 

        #Epoch processing block:
        x,hn = self.epochrnn(x)
        x    = self.seqDropout(x)


        #self-attention:
        v      = torch.tanh(torch.matmul(torch.reshape(x, [-1, self.nHidden*2]), self.attweight_w) + torch.reshape(self.attweight_b, [1, -1]))
        vu     = torch.matmul(v, torch.reshape(self.attweight_u, [-1, 1]))
        exps   = torch.reshape(torch.exp(vu), [-1, self.timeBins])
        alphas = exps / torch.reshape(torch.sum(exps, 1), [-1, 1])
        x      = torch.sum(x * torch.reshape(alphas, [-1, self.timeBins, 1]), 1)

        
        #sequence processing block
        x     = x.reshape(-1, self.L, self.nHidden*2)
        x, hn = self.seqRnn(x)
        x     = self.seqDropout(x)

        #return to epochs:
        x = x.reshape(-1, self.nHidden*2) # flatten (L*B, nHidden*2)
        
        # the hidden layer before it is fed to the fully connected layer
        before_fc = x

        
        #out:
        x = self.fc(x) # reshaped to (100, 5) from (100, 128)

        return x, before_fc
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.base_lr,
                                    weight_decay=self.weight_decay)
        
        # if schedule and arguments provided, use them. Otherwise use constant base_lr
        if self.lr_sched and self.sched_args:
            scheduler = self.lr_sched(optimizer, **self.sched_args)
        else:
            if self.sched_args:
                print("Arguments for schedule provided, but no schedule. A schedule will not be used.", file=sys.stderr)
            identity_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda epoch: self.base_lr])
            scheduler = identity_schedule
    
        return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler,
        'monitor': 'valKappa'
         }
    
    def epochMetrics(self,epochOutputs):
        epochPreds=[]
        trueLabels=[]
        totLoss=0
        for out in epochOutputs:
            epochPreds=np.append(epochPreds,out['pred_labels'].cpu())
            trueLabels=np.append(trueLabels,out['labels'].cpu())
            totLoss+=out['loss'].cpu()
        
        totLoss/=trueLabels.size 
        kappa=np.around(cohen_kappa_score(epochPreds,trueLabels),4)
        acc=np.mean(epochPreds==trueLabels)
        
        return totLoss, kappa, acc
    
    def training_step(self, batch, batch_idx):
        xtemp, _, ytemp = batch            
        y_pred, _ = self(xtemp)
        loss = F.cross_entropy(y_pred,ytemp)
        a,pred_labels=torch.max(y_pred,1) #b is 0-4
        if self.lr_schedulers():
            self.lr_schedulers().step() #update lr
        return {'loss':loss,'pred_labels':pred_labels,'labels':ytemp,'idx':batch_idx}
    
    def training_epoch_end(self, training_step_outputs):
        totLoss,kappa,acc=self.epochMetrics(training_step_outputs)
        self.log("lr", self.lr_schedulers().get_last_lr()[0])
        self.log('trainLoss',totLoss)
        self.log('trainKappa',kappa)

       
    def validation_step(self, batch, batch_idx):
        xtemp, _, ytemp = batch            
        y_pred, _ = self(xtemp)
        loss = F.cross_entropy(y_pred,ytemp)
        a,pred_labels=torch.max(y_pred.cpu(),1) #b is 0-4
        return {'loss':loss,'pred_labels':pred_labels,'labels':ytemp,'idx':batch_idx }
    
    def validation_epoch_end(self, validation_step_outputs):
        totLoss,kappa,acc=self.epochMetrics(validation_step_outputs)
        self.log('valLoss',totLoss)
        self.log('valKappa',kappa)
        
    def test_step(self, batch, batch_idx):
        xtemp, idx = batch    
        y_pred, _ = self(xtemp)

        return {'loss':None,'y_pred':y_pred,'idx':idx}

    
    def test_epoch_end(self, test_step_outputs):
        #get dimensions:
        nRows=0
        for out in test_step_outputs:
            nRows+=len(out['idx'])

        
        y_pred=np.zeros((nRows,5))
        idxs=[]
        for out in test_step_outputs:
            y_pred[out['idx'].cpu().numpy().astype(int),:]=out['y_pred'].cpu()
            idxs=np.append(idxs,out['idx'].cpu())

        results = {'y_pred':y_pred,'idxs':idxs, 'progress_bar': idxs}
        self.test_results = results
        return results
        
   

if __name__ == '__main__':
    
    net=SeqSleepNetBase(3)
    c = torch.randn(1,189,129,1)
    x=torch.randn(3*net.L,29,129,1)
    x[1,:,2,:]=1
    output=net(x,c)
    print('unit test complete')
