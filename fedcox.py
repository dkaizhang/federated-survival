import copy
import numpy as np
import torch
import torch.nn.functional as F

from dataset import DatasetSplit, sample_iid

from tensorboardX import SummaryWriter
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

rng = np.random.default_rng(123)



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


