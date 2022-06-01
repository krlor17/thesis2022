import pytorch_lightning as pl
import torch

class RandomPlaceboEncoder(pl.LightningModule):
    """ A Placebo encoder outputting a random [0,1] vector with specified dim.
    """
    def __init__(self, latent_dim, cal_shape, L=20):
        """
        Construct a random placebo encoder.

        Parameters
        ----------
        latent_dim : int
            Latent dimensionality of encoder.
            
        Returns
        -------
        RandomPlaceboEncoder

        """
        super().__init__()
        self.latent_dim = latent_dim
        self.cal_shape = cal_shape
        self.L = L
        self.dev = torch.cuda.current_device()
        
    def forward(self,cal):
        cal = cal.reshape((-1,) + self.cal_shape ) # (batch, T, F)
        batch_size = cal.shape[0]
        latent = torch.rand((self.L*batch_size, self.latent_dim)) # (L*B, latent_dim) 
        return latent.to(self.dev)