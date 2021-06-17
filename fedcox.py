import numpy as np
import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

rng = np.random.default_rng(123)

class DenseBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, batch_norm=True, dropout=0, activation=nn.ReLU, 
                    w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias)
        # fill in self.linear.weight.data with kaiming normal
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(dim_out) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input

class MLP(nn.Module):
    def __init__(self, dim_in, num_nodes, dim_out, batch_norm=True, dropout=None, activation=nn.ReLU,
                output_activation=None, output_bias=True,
                w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        nodes = [dim_in].append(num_nodes)
        net = []
        for d_in, d_out in zip(nodes[:-1], nodes[1:]):
            net.append(DenseBlock(d_in, d_out, bias=True, batch_norm=batch_norm, 
                        dropout=dropout, activation=activation, w_init_=w_init_))
        net.append(nn.Linear(num_nodes[-1], dim_out, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)

# scale the loss to make it data size invariant
def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', or 'mean'.")

def negative_llh(phi: Tensor, idx_durations: Tensor, events: Tensor, 
                reduction: str = 'mean') -> Tensor:
    """
    Arguments:
        phi -- N x T. Estimates in (-inf, inf) where hazard = sigmoid(phi). 
        idx_durations -- T. Discretised event times represent as indices.
        events -- T. Indicator of event (1.) or censoring (0.). Floats.
    Returns:
        Negative LLH as a 1dim Tensor
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    # stack events and idx_durations to get Tx1 Tensor
    # .view(-1, 1) -- infer first dimension to get second dimension as size=1 
    events = events.view(-1, 1)    
    idx_durations = idx_durations.view(-1, 1)
    # create NxT zero Tensor and put in each 
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)
    # with logits to pass phi through a sigmoid first -- recall it's unconstrained
    bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')
    # sum the loss until failure time:
        # first sum until each time
        # then gather the summed loss at relevant failure time
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)


def sample_iid(data, num_centers):
    """
    Randomly split data evenly across each centre
    Arguments:
    data -- combined data
    num_centres -- number of centres to spread over    
    Returns:
    Dict with centre_id : indices of data assigned to centre
    """
    num_items = int(len(data)/num_centers)
    dict_center_idxs, all_idxs = {}, [i for i in range(len(data))]
    for i in range(num_centers):
        dict_center_idxs[i] = set(rng.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_center_idxs[i])
    return dict_center_idxs    

def make_dataloader(data, batch_size, shuffle, num_workers=0, to_tensor=True):
    pass



class Member():
    def __init__(self, data, idxs: set) -> None:
        pass
    


class Federation():

    def __init__(self, net, num_centers, logger=None, loss=None, optimizer=None, device=None):
        self.global_model = net
        self.num_centers = num_centers
        self.logger = logger
        self.loss = loss
        self.optimizer = optimizer
        self.set_device(device)

    def device(self):
        return self._device

    def set_device(self, device):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device
        self.global_model.to(self.device())


    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224,
            **kwargs):
        pass

