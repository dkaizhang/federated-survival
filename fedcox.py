import torch
import torchtuples as tt

from pycox.models.loss import CoxPHLoss
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset



class FedCox(tt.Model):

    def __init__(self, net, loss=None, optimizer=None, device=None):
        super().__init__(net, loss, optimizer, device)
    
    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224,
            **kwargs):

        self.training_data = tt.tuplefy(input, target)
        return super().fit(input, target, batch_size, epochs, callbacks, verbose,
                           num_workers, shuffle, metrics, val_data, val_batch_size,
                           **kwargs)
    
    
