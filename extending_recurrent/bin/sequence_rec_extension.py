#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: krlor17
"""
import sys
import pytorch_lightning as pl
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from filterbank_shape import FilterbankShape 

from sklearn.metrics import cohen_kappa_score
from extended_loader import TestDataset, test_collate_fn

class SequenceExtension(pl.LightningModule):
    def __init__(self, encoder, base_lr=1e-3,lr_sched=None, sched_args=None, weight_decay=0.0, dropout_rate=0.0, log_every_step = False, L=20):
        super().__init__()
        assert encoder.latent_dim # encoder must have a latent_dim attribute
        self.encoder = encoder  # Encoder to get latent space from calibration data
        self.latent_dim = encoder.latent_dim
        self.L = L
        
        self.log_every_step = log_every_step 
        self.base_lr = base_lr
        self.lr_sched = lr_sched
        self.sched_args = sched_args
        self.weight_decay = weight_decay
        self.save_hyperparameters(logger=False)
        
        # base hyperparameters
        self.nHidden=64
        self.nFilter=32
        self.attentionSize=64
        self.dropout_rate=dropout_rate
        self.timeBins=29

        #-------------------------base:filterbank------------------------------
        filtershape = FilterbankShape()
        
        #triangular filterbank shape
        shape=torch.tensor(filtershape.lin_tri_filter_shape(nfilt=self.nFilter,
                                                            nfft=256,
                                                            samplerate=100,
                                                            lowfreq=0,
                                                            highfreq=50),dtype=torch.float)
        
        self.Wbl = nn.Parameter(shape,requires_grad=False)
        #filter weights:
        self.Weeg = nn.Parameter(torch.randn(self.nFilter,1))
        
        #--------------------------------base----------------------------------

        # epoch-wise
        self.epochrnn = nn.GRU(self.nFilter,self.nHidden,1,bidirectional=True,batch_first=True)

        #attention-layer:       
        self.attweight_w = nn.Parameter(torch.randn(2*self.nHidden, self.attentionSize))
        self.attweight_b = nn.Parameter(torch.randn(self.attentionSize))
        self.attweight_u = nn.Parameter(torch.randn(self.attentionSize))
        
        #epoch sequence block:
        self.seqDropout = torch.nn.Dropout(self.dropout_rate, inplace=False)
        self.seqRnn = nn.GRU(self.nHidden*2 + encoder.latent_dim, self.nHidden, 1, bidirectional=True, batch_first=True)
        
        #output:
        self.fc = nn.Linear(2*self.nHidden,5)

    def forward(self, x, c):
        """
        Forward computation of network.

        Parameters
        ----------
        x : pytorch tensor 
            Time series data fed to SeqSleepNet base model
        c : pytorch tensor
            Calibration data fed to encoder

        Returns
        -------
        Output pred pre-softmax

        """
        assert (x.shape[0]/self.L).is_integer() # we need to pass a multiple of L epochs
        assert (x.shape[1]==self.timeBins)
        assert (x.shape[2]==129)
        
        
        #encode calibration data
        latent = self.encoder(c) # L*batch_size, latent_dim
        batch_size = latent.shape[0]//self.L
        #forward run w. n
        
        x = x.permute([0,3,1,2])  # switch ordering to (Batch, Time, Freq, Channel)

        #filtering:
        Wfb = torch.multiply(torch.sigmoid(self.Weeg[:,0]),self.Wbl)
        x   = torch.matmul(x, Wfb) # filtering
        x   = x.reshape( [-1, self.timeBins, self.nFilter])
        
        #Epoch processing block:
        x,hn = self.epochrnn(x)
        x    = self.seqDropout(x)

        #self-attention:
        v      = torch.tanh(torch.matmul(torch.reshape(x, [-1, self.nHidden*2]), self.attweight_w) + torch.reshape(self.attweight_b, [1, -1]))
        vu     = torch.matmul(v, torch.reshape(self.attweight_u, [-1, 1]))
        exps   = torch.reshape(torch.exp(vu), [-1, self.timeBins])
        alphas = exps / torch.reshape(torch.sum(exps, 1), [-1, 1])
        x      = torch.sum(x * torch.reshape(alphas, [-1, self.timeBins, 1]), 1)
        
        # concatenate
        x = torch.cat((x,latent), dim=1)
        
        #sequence processing block
        x     = x.reshape(-1, self.L, self.nHidden*2 + self.encoder.latent_dim)
        x, hn = self.seqRnn(x)
        x     = self.seqDropout(x)

        #return to epochs:
        x = x.reshape(-1, self.nHidden*2) # flatten (L*B, nHidden*2)
        
        #out:
        x = self.fc(x) # reshaped to (100, 5) from (100, 128)

        return x
    
    def configure_optimizers(self):
        scheduler = None
        
        # if schedule and arguments provided, use them. Otherwise use constant base_lr
        if self.lr_sched and self.sched_args:
            optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.base_lr,
                                    weight_decay=self.weight_decay)
            scheduler = self.lr_sched(optimizer, **self.sched_args)
        else:
            optimizer = torch.optim.Adam(self.parameters(),
                                    lr=self.base_lr,
                                    weight_decay=self.weight_decay)
            if self.sched_args:
                print("Arguments for schedule provided, but no schedule. A schedule will not be used.", file=sys.stderr)
            identity_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda epoch: self.base_lr])
            scheduler = identity_schedule
    
        return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler,
        'monitor': 'valKappa'
         }
    
    def _step(self, batch, batch_idx, prefix):
        x, c, y = batch            
        c = c.float()
        y_pred = self(x, c)
        loss = F.cross_entropy(y_pred,y)
        _,pred_labels=torch.max(y_pred,1)
        if prefix == "train" and self.lr_schedulers():
            self.lr_schedulers().step() #update lr
        return {'loss':loss,'pred_labels':pred_labels,'labels':y,'idx':batch_idx }

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        x, c, idx = batch    
        y_pred = self(x,c)
        return {'loss':None,'y_pred':y_pred,'idx':idx}


    def _epoch_metrics(self,epoch_steps, prefix):
        epoch_preds = []
        true_labels = []
        tot_loss = 0
        for step in epoch_steps:
            epoch_preds = np.append(epoch_preds,step['pred_labels'].cpu())
            true_labels = np.append(true_labels,step['labels'].cpu())
            tot_loss += step['loss'].cpu()
        
        tot_loss /= true_labels.size 
        kappa = np.around(cohen_kappa_score(epoch_preds,true_labels),6)
        acc = np.mean(epoch_preds == true_labels)
      
        self.log(prefix+'Loss', tot_loss)
        self.log(prefix+'Accuracy',acc)
        self.log(prefix+'Kappa', kappa )
        if self.lr_schedulers():
            self.log("lr",self.lr_schedulers().get_last_lr()[0])
        
    
    def training_epoch_end(self, training_step_outputs):
        self._epoch_metrics(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        self._epoch_metrics(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        #get dimensions:
        num_rows=0
        for out in test_step_outputs:
            num_rows+=len(out['y_pred'])

        
        y_pred=np.zeros((num_rows,5))
        idxs=[]
        idx = 0
        for out in test_step_outputs:
            batch_size = out["y_pred"].shape[0]
            batch_idxs = np.array(range(idx,idx+batch_size))
            idx += batch_size
            y_pred[batch_idxs,:] = out['y_pred'].cpu()
            idxs=np.append(idxs,out['idx'].cpu())

        results = {'y_pred':y_pred,'idxs':idxs, 'progress_bar': idxs}
        self.test_results = results
        return results

from constant_placebo import ConstantPlaceboEncoder
if __name__ == '__main__':
    cuda = torch.cuda.current_device()
    encoder = ConstantPlaceboEncoder(16, (129,188))
    encoder.to(cuda)
    net=EpochwiseExtension(encoder)
    net.to(cuda)
    c = torch.randn(32*1,188,129,1).to(cuda)
    x=torch.randn(32*net.L,29,129,1).to(cuda)
    x[1,:,2,:]=1
    output=net(x,c)
    print('unit test complete')
