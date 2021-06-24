import copy
import numpy as np
import torch

from dataset import DatasetSplit, sample_iid
from loss import negative_llh

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

rng = np.random.default_rng(123)

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
    def __init__(self, optimizer, lr, features, labels, split, idxs: set, epochs, logger, loss, batch_size=256, device=None, verbose=True ) -> None:
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.logger = logger
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.loss = loss
        self.trainloader, self.validloader, self.testloader = self.train_val_test(features, labels, split, list(idxs))


    def train_val_test(self, features, labels, split, idxs: list):
        # rng.shuffle(idxs)
        idxs_train = idxs[:int(split[0]*len(idxs))]
        idxs_val = idxs[int(split[0]*len(idxs)):int((split[0] + split[1])*len(idxs))]
        idxs_test = idxs[int((split[0] + split[1])*len(idxs)):]
        trainloader = DataLoader(DatasetSplit(features, labels, idxs_train),
                                 batch_size=self.batch_size, shuffle=True)
        validloader = DataLoader(DatasetSplit(features, labels, idxs_val),
                                 batch_size=max(1,int(len(idxs_val)/10)), shuffle=False)
        testloader = DataLoader(DatasetSplit(features, labels, idxs_test),
                                batch_size=max(1,int(len(idxs_test)/10)), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        else:
            ValueError

        for iter in range(self.epochs):
            batch_loss = []
            for batch_idx, data in enumerate(self.trainloader):
                features, durations, events = data[0], data[1], data[2]
                features, durations, events = features.to(self.device), durations.to(self.device), events.to(self.device)

                model.zero_grad()
                phi = model(features)
                loss = self.loss(phi, durations, events)
                loss.backward()
                optimizer.step()

                if self.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


class Federation():
    """
    Accepts a neural net and performs the training with the nll as loss function (set in Member)
    """
    def __init__(self, net, num_centers, optimizer, lr, local_epochs=1, loss=negative_llh,  device=None, logger=SummaryWriter('./logs')):
        self.global_model = net
        self.num_centers = num_centers
        self.optimizer = optimizer
        self.lr = lr
        self.local_epochs = local_epochs
        self.loss = loss
        self.set_device(device)
        self.logger = logger

    def device(self):
        return self._device

    def set_device(self, device):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device
        self.global_model.to(self.device())

    def fit(self, features, labels, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224,
            print_every=2):
        self.global_model.train()
        dict_center_idxs = sample_iid(features, self.num_centers)

        train_loss = []

        for epoch in range(epochs):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')

            for center_idx in range(self.num_centers):
                member = Member(self.optimizer, self.lr, features, labels, [0.8, 0.1, 0.1], dict_center_idxs[center_idx], self.local_epochs, self.logger, self.loss)
                w, loss = member.update_weights(model=copy.deepcopy(self.global_model), global_round=epoch) 
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
            global_weights = average_weights(local_weights)

            #update global model with new weights
            self.global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
