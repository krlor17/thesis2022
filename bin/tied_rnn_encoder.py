import sys
import os
import pytorch_lightning as pl
import numpy as np
import scipy
import torch
import torch.nn as nn

class TiedRNNEncoder(pl.LightningModule):
    """ An Encoder using the same filterbank as the base model 
        and a GRU RNN to produce a latent representation
    """
    def __init__(self, latent_dim, cal_shape, base, dropout_rate=0.0, L=20):
        """
        Construct a tied RNN encoder.

        Parameters
        ----------
        latent_dim : int
            Latent dimensionality of encoder. Must be divisible by 2.
        cal_shape : tuple
            Dimensions of calibration spectrogram.
        dropout : float
            Dropout rate.
        base : SeqSleepNetBase
            Base model to tie filterbank with.

        Returns
        -------
        None.

        """
        assert float.is_integer(latent_dim/2)
        assert type(cal_shape) == tuple
        assert len(cal_shape) == 2
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = latent_dim//2
        self.base = base # base model for tied filterbank
        self.cal_shape = cal_shape
        self.L = L
        
        # bidirectional GRU rnn w. hidden state of size latent_dim -> output 2*latent_dim
        self.rnn = nn.GRU( input_size=self.base.nFilter,
                          hidden_size=self.hidden_dim,
                          num_layers=1,
                          bidirectional=True,
                          batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm(normalized_shape=[self.latent_dim])
        # self.save_hyperparameters()
        
    def forward(self,cal):
        cal = cal.reshape((-1,) + self.cal_shape ) # (batch, F, T)
        cal = cal.permute([0,2,1]) # (batch, T, F)
        batch_size = cal.shape[0]
        # Filterbank
        Wfb = torch.multiply(torch.sigmoid(self.base.Weeg[:,0]),self.base.Wbl)
        cal = torch.matmul(cal, Wfb)
        # RNN 
        cal = torch.reshape(cal, (-1,self.cal_shape[1],self.base.nFilter))  # (batch, T, M)
        _, hn = self.rnn(cal)                                               # hn (2, B, hidden_dim)
        latent = hn.reshape((-1,self.latent_dim))                           # (B, latent_dim )
        latent = self.layer_norm(latent)
        latent = self.dropout(latent)
        # (B, latent_dim) -> (L*B, latent_dim) st. each latent representation is return L *consecutive* times
        latent = latent.reshape(batch_size,1, self.latent_dim)
        latent = latent.expand(batch_size, self.L, self.latent_dim) # expand to match flattened base layer
        latent = latent.reshape(self.L*batch_size, self.latent_dim) # (L*B, latent_dim) 
        return latent

# unit test
from seqsleepnet_base import SeqSleepNetBase
if __name__ == "__main__":
    print("~~~ Unit test ~~~")
    base = SeqSleepNetBase()
    encoder = TiedRNNEncoder(16, (129,188), base)
    test_input = torch.rand((1,129,188))
    out = encoder(test_input)
    unique = torch.unique(out, dim=0)
    print("Output shape:",out.shape)
    print("Number of unique vectors:", len(unique))
    print("Unique encodings:", unique.detach().numpy())
    if len(unique) == 1 and out.shape[0] == 20 and out.shape[1] == 16:
        print("Success.")
    else:
        print("Failure.")