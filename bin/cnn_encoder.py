import sys
import os
import pytorch_lightning as pl
import numpy as np
import scipy
import torch
import torch.nn as nn

def _size_reduction(in_size, number_blocks=4, number_conv=2 ):
    """
    Compute the output size after applying a specified number of ccm blocks.

    Parameters
    ----------
    in_size : int
        Size before applying ccm blocks.
    number_ccm_blocks : int, optional.
        Number of ccm blocks applied. Default is 4.

    Returns
    -------
    int

    """
    out_size = in_size
    for _ in range(number_blocks):
        out_size = (out_size - number_conv  - 2)//2 + 1
    return out_size

class VGGBlock(nn.Module):
    """ A conv-conv-maxpool block as described for VGG16 """
    def __init__(self, channels_in, channels_out, dropout_rate, use_relu=True):
        """
        Construct a block of 2 convolutional layers w. 3x3 kernels
        and a 2x2 maxpooling layer.

        Parameters
        ----------
        filters_in : int
            Number of input channels
        filters_out : int
            Number of output channels i.e. filters in convolutional layers
        dropout : float
            Dropout rate for spatial dropout i.e. rate of channels dropped.
        """
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.dropout_rate = dropout_rate
        self.use_relu = use_relu
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self.conv1 = nn.Conv2d(channels_in, channels_out, (3,3))
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.bn = nn.BatchNorm2d(channels_out)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.maxpool(x)
        x = self.bn(x)
        return x

class CNNEncoder(pl.LightningModule):
    """ A VGG inspired CNN encoder """
    def __init__(self, latent_dim, cal_shape, dropout_rate, L=20):
        super().__init__()
        self.L = 20
        self.latent_dim = latent_dim
        self.cal_shape = cal_shape
        self.dropout_rate = dropout_rate
        self.ccm1 = VGGBlock(1,16, dropout_rate)
        self.ccm2 = VGGBlock(16,32, dropout_rate)
        self.ccm3 = VGGBlock(32,64, dropout_rate)
        self.ccm4 = VGGBlock(64,64, dropout_rate)
        self.last_height = _size_reduction(cal_shape[0])
        self.last_width = _size_reduction(cal_shape[1])        
        self.fc = nn.Linear(self.last_height*self.last_width*64, latent_dim)
        self.ln = nn.LayerNorm([latent_dim])
        
    def forward(self, c):
        c = c.reshape((-1,1,self.cal_shape[0],self.cal_shape[1]))
        batch_size = c.shape[0]
        c = self.ccm1(c)
        c = self.ccm2(c)
        c = self.ccm3(c)
        c = self.ccm4(c)
        c = c.reshape((-1, self.last_height*self.last_width*64))
        latent = self.fc(c)
        latent = self.ln(latent)
        # expand to match flattened base layer
        latent = latent.reshape(batch_size,1, self.latent_dim)
        latent = latent.expand(batch_size, self.L, self.latent_dim)
        latent = latent.reshape(self.L*batch_size, self.latent_dim) # (L*batch size, latent_dim)
        return latent


if __name__ == "__main__":
    print("~~~ Unit test ~~~")
    encoder = CNNEncoder(16, (129,188), dropout_rate=0.1, L=20)
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