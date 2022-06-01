import pytorch_lightning as pl
import torch

class ConstantPlaceboEncoder(pl.LightningModule):
    """ A Placebo encoder outputting a constant all-ones vector with specified dim.
    """
    def __init__(self, latent_dim, cal_shape, L=20):
        """
        Construct a constant placebo encoder.

        Parameters
        ----------
        latent_dim : int
            Latent dimensionality of encoder.
            
        Returns
        -------
        ConstantPlaceboEncoder

        """
        super().__init__()
        self.latent_dim = latent_dim
        self.L = L
        self.cal_shape = cal_shape
        self.dev = torch.cuda.current_device()
        
    def forward(self,cal):
        cal = cal.reshape((-1,) + self.cal_shape ) # (batch, T, F)
        batch_size = cal.shape[0]
        latent = torch.ones((self.L*batch_size, self.latent_dim)) # (L*B, latent_dim) 
        return latent.to(self.dev)                        