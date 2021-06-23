import copy
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
        idx_durations -- N. Event times for each individual represented as indices based on discretisation.
        events -- N. Indicator of event (1.) or censoring (0.) for each individual. Floats.
    Returns:
        Negative LLH as a 1dim Tensor
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    # stack events and idx_durations to get Nx1 Tensor
    # .view(-1, 1) -- infer first dimension to get second dimension as size=1 
    events = events.view(-1, 1)    
    idx_durations = idx_durations.view(-1, 1)
    # create NxT zero Tensor and put in each row at the idx_duration whether the event occurred
    # censored ppl have row of zeros, uncensored ppl have row with a single one
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


def average_weights(w):
    """
    Arguments:
    w -- state dict of a model.
    Returns:
    w_avg -- the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class Discretiser():
    def __init__(self, cuts):
        self._cuts = cuts
        self.cuts = None

    def fit(self, durations):
        durations = durations.astype(np.float64)
        self.cuts = np.linspace(0, durations.max(), self._cuts, dtype=np.float64)
        return self
    
    def transform(self, durations, events):    
        idx_durations = np.digitize(durations, np.concatenate((self.cuts, [np.infty]), dtype=np.float64)) - 1 + events
        return (idx_durations, events)

    def fit_transform(self, durations, events):
        self.fit(durations)
        durations, events = self.transform(durations ,events)
        return (durations, events)

    @property
    def dim_out(self):
        if self.cuts is None:
            raise ValueError("Need to call `fit` before this is accessible.")
        return len(self.cuts)




class DatasetSplit(torch.utils.data.Dataset):
    # idxs in original data which belong to center 
    def __init__(self, features, labels, idxs):
        self.features = features
        self.durations = labels[0]
        self.events = labels[1]
        # center-specific indices which map to idxs
        # e.g. [0,1,5] belong to center -> [0,1,2] internally
        self.idxs = [int(i) for i in idxs]

    def __getitem__(self, index):
        internal_idx = self.idxs[index]
        return self.features[internal_idx], self.durations[internal_idx], self.events[internal_idx]
       
    def __len__(self):
        return len(self.idxs)



class Member():
    def __init__(self, features, labels, split, idxs: set, epochs, logger, batch_size=256, optimizer=None, device=None, verbose=True ) -> None:
        self.trainloader, self.validloader, self.testloader = self.train_val_test(features, labels, split, list(idxs))
        self.epochs = epochs
        self.logger = logger
        self.batch_size = batch_size
        self.device = device
        self.optimizer = optimizer
        self.verbose = verbose

        self.loss = negative_llh


    def train_val_test(self, features, labels, split, idxs: list):
        rng.shuffle(idxs)
        idxs_train = idxs[:int(split[0]*len(idxs))]
        idxs_val = idxs[int(split[0]*len(idxs)):int((split[0] + split[1])*len(idxs))]
        idxs_test = idxs[int((split[0] + split[1])*len(idxs)):]
        trainloader = DataLoader(DatasetSplit(features, labels, idxs_train),
                                 batch_size=self.batch_size, shuffle=True)
        validloader = DataLoader(DatasetSplit(features, labels, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(features, labels, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []

        for iter in range(self.epochs):
            batch_loss = []
            for batch_idx, data in enumerate(self.trainloader):
                features, durations, events = data[0], data[1], data[2]
                features, durations, events = features.to(self.device), durations.to(self.device), events.to(self.device)

                model.zero_grad()
                phi = model(features)
                loss = self.loss(phi, durations, events)
                loss.backward()
                self.optimizer.step()

                if self.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


# Owns Members
class Federation():
    def __init__(self, net, num_centers, local_epochs=1, logger=None, loss=None, optimizer=None, device=None):
        self.global_model = net
        self.num_centers = num_centers
        self.local_epochs = local_epochs
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


    def fit(self, features, labels, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224,
            **kwargs):
        self.global_model.train()
        dict_center_idxs = sample_iid(self.features, self.num_centers)

        for epoch in epochs:
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')

            for center_idx in range(self.num_centers):
                member = Member(features, labels, [0.8, 0.1, 0.1],dict_center_idxs[center_idx], self.local_epochs, self.logger)
                w, loss = member.update_weights(model=copy.deepcopy(self.global_model), global_round=epoch) 
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)


