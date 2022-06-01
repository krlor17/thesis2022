import sys
import os
import math
import pytorch_lightning as pl
import numpy as np
import scipy
import torch
import torch.nn as nn


class SimpleAlphaEncoder(pl.LightningModule):
    """ 
        An Encoder outputting a simple sum of alpha-band activity for 
        3 parts of the calibration signal:
        1) 1. interval of closed eyes, 
        2) intermediate interval of open eyes
        3) 2. interval closed eyes.
        The encoder assumes that such 3 consecutive intervals can be extracted from
        calibration data.
    """
    def __init__(self, cal_shape, timemarks, nfft=256, freq_span=50, L=20):
        # assert cal_shape[0] == nfft//2 + 1
        assert len(cal_shape) == 2
        assert len(timemarks) == 5
        assert timemarks == sorted(timemarks)
        assert timemarks[0] >= 0
        assert timemarks[4] <= cal_shape[0]
        super().__init__()
        self.cal_shape = cal_shape
        self.start = timemarks[0]
        self.close1 = timemarks[1]
        self.open = timemarks[2]
        self.close2 = timemarks[3]
        self.end = timemarks[4]
        self.L = L
        self.latent_dim = 4     # for compatibility with ExploratoryExtension
        
        n_freq_bins = nfft//2 + 1
        alpha_lower = 4
        alpha_upper = 8
        self.band_lower = math.floor(n_freq_bins/freq_span*alpha_lower)
        self.band_upper = math.ceil(n_freq_bins/freq_span*alpha_upper)
        self.ln = nn.LayerNorm(normalized_shape=[self.latent_dim])
        self.save_hyperparameters()
    
    def forward(self, c):
        c = c.reshape((-1,)+self.cal_shape)
        batch_size = c.shape[0]

        sum_open1 =torch.sum(c[:, self.band_lower:self.band_upper, self.start:self.close1], dim=[1,2])
        sum_closed_1 =torch.sum(c[:,self.band_lower:self.band_upper,self.close1:self.open], dim=[1,2])
        sum_open2 =torch.sum(c[:, self.band_lower:self.band_upper, self.open:self.close2], dim=[1,2])
        sum_closed_2 =torch.sum(c[:, self.band_lower:self.band_upper, self.close2:self.end], dim=[1,2])
        sum_open1 = sum_open1.reshape((-1,1))
        sum_closed_1 = sum_closed_1.reshape((-1,1))
        sum_open2 = sum_open2.reshape((-1,1))
        sum_closed_2 = sum_closed_2.reshape((-1,1))
        latent = torch.concat((sum_open1,sum_closed_1, sum_open2, sum_closed_2), dim=1)
        latent = self.ln(latent) ##normalize
        latent = latent.reshape(batch_size,1, self.latent_dim)
        latent = latent.expand(batch_size, self.L, self.latent_dim) # expand to match flattened base layer
        latent = latent.reshape(self.L*batch_size, self.latent_dim) # (L*B, latent_dim) 
        return latent
    
class AlphaEncoder(pl.LightningModule):
    """ 
        An Encoder outputting a simple sum of alpha-band activity for 
        3 parts of the calibration signal:
        1) 1. interval of closed eyes, 
        2) intermediate interval of open eyes
        3) 2. interval closed eyes.
        The encoder assumes that such 3 consecutive intervals can be extracted from
        calibration data.
    """
    def __init__(self, latent_dim, cal_shape, timemarks, nfft=256, freq_span=50, L=20):
        assert len(cal_shape) == 2
        assert len(timemarks) == 2
        assert timemarks == sorted(timemarks)
        assert timemarks[0] >= 0
        assert timemarks[-1] <= cal_shape[0]
        super().__init__()
        self.cal_shape = cal_shape
        self.start = timemarks[0]
        self.end = timemarks[1]
        
        self.latent_dim = latent_dim
        self.L = 20
        
        n_freq_bins = nfft//2 + 1
        alpha_lower = 4
        alpha_upper = 8
        self.band_lower = math.floor(n_freq_bins/freq_span*alpha_lower)
        self.band_upper = math.ceil(n_freq_bins/freq_span*alpha_upper)
        
        self.fc = nn.Linear(in_features=timemarks[1]-timemarks[0], out_features=self.latent_dim)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(normalized_shape=[self.latent_dim])
        self.save_hyperparameters()
    
    def forward(self, c):
        c = c.reshape((-1,)+self.cal_shape)
        batch_size = c.shape[0]
        # sum along frequency dim
        cut = torch.sum(c[:,self.band_lower:self.band_upper,self.start:self.end], dim=1)
        latent = self.fc(cut)
        latent = self.relu(latent)
        latent = self.ln(latent)
        latent = latent.reshape(batch_size,1, self.latent_dim)
        latent = latent.expand(batch_size, self.L, self.latent_dim) # expand to match flattened base layer
        latent = latent.reshape(self.L*batch_size, self.latent_dim) # (L*B, latent_dim) 
        return latent
    
if __name__ == "__main__":
    print("Unit test: SimpleAlphaEncoder")
    encoder1 = SimpleAlphaEncoder(cal_shape=(129,188), timemarks=[1,2,3,4,5])
    test_input = torch.rand((1,129,188))
    out = encoder1(test_input)
    unique = torch.unique(out, dim=0)
    print("Output shape:",out.shape)
    print("Number of unique vectors:", len(unique))
    print("Unique encodings:", unique.detach().numpy())
    if len(unique) == 1 and out.shape[0] == 20 and out.shape[1] == 4:
        print("Success.")
    else:
        print("Failure.")
    print("")
    print("Unit test: AlphaEncoder")        
    encoder2 = AlphaEncoder(16,  cal_shape=(129,188), timemarks=[1,2])
    test_input = torch.rand((1,129,188))
    out = encoder2(test_input)
    unique = torch.unique(out, dim=0)
    print("Output shape:",out.shape)
    print("Number of unique vectors:", len(unique))
    print("Unique encodings:", unique.detach().numpy())
    if len(unique) == 1 and out.shape[0] == 20 and out.shape[1] == 16:
        print("Success.")
    else:
        print("Failure.")