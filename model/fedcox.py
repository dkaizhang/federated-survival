import copy
import numpy as np
import pandas as pd
import torch

from model.dataset import DatasetSplit, sample_by_quantiles, sample_iid
from model.loss import negative_llh

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
    def __init__(self, optimizer, lr, features, labels, split, idxs: set, epochs, logger, loss, batch_size=256, device=None) -> None:
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.logger = logger
        self.batch_size = batch_size
        self.device = device
        self.loss = loss
        self.train_losses = []
        self.val_losses = []
        print(len(list(idxs)))
        self.trainloader, self.validloader, self.testloader = self.train_val_test(features, labels, split, list(idxs))


    def get_train_val_test(self):
        return self.trainloader, self.validloader, self.testloader

    def train_val_test(self, features, labels, split, idxs: list):
        # rng.shuffle(idxs)
        idxs_train = idxs[:int(split[0]*len(idxs))]
        idxs_val = idxs[int(split[0]*len(idxs)):int((split[0] + split[1])*len(idxs))]
        idxs_test = idxs[int((split[0] + split[1])*len(idxs)):]
        trainloader = DataLoader(DatasetSplit(features, labels, idxs_train),
                                 batch_size=self.batch_size, shuffle=True)
        validloader = DataLoader(DatasetSplit(features, labels, idxs_val),
                                 batch_size=int(self.batch_size/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(features, labels, idxs_test),
                                batch_size=int(self.batch_size/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, verbose):
        model.to(self.device)
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

                if verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            self.val_losses.append(self.get_loss(model, validate=True))
            self.train_losses.append(sum(epoch_loss) / len(epoch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def get_loss(self, model, validate=True):
        model.to(self.device)
        model.eval()
        loader = self.validloader if validate else self.testloader
        batch_loss = []
        for batch_idx, data in enumerate(loader):
            features, durations, events = data[0], data[1], data[2]
            features, durations, events = features.to(self.device), durations.to(self.device), events.to(self.device)
            phi = model(features)
            loss = self.loss(phi, durations, events)
            batch_loss.append(loss.item())
        model.train()
        return sum(batch_loss) / len(batch_loss)

    def predict_hazard(self, model, input=None):
        model.to(self.device)
        model.eval()
        loader = input if not None else self.testloader 
        hazard = torch.cat([model(data[0].to(self.device)).sigmoid() for data in loader], axis=0)         
        model.train()
        return hazard

    def predict_surv(self, model, input=None):
        hazard = self.predict_hazard(model, input)
        surv = (1 - hazard).log().cumsum(1).exp()
        return surv.cpu().detach().numpy()

    def predict_surv_df(self, model, cuts, input=None):
        surv = self.predict_surv(model, input)
        return pd.DataFrame(surv.transpose(), cuts)

class Federation():
    """
    Accepts data, creates Members and distributes data
    Accepts a neural net and performs the training with the nll as loss function (set in Member)
    """
    def __init__(self, features, labels, net, num_centers, optimizer, lr, stratify_on=None, stratify_labels=False, batch_size=256, local_epochs=1, loss=negative_llh,  device=None, logger=SummaryWriter('./logs')):
        self.features= features
        self.labels= labels
        self.global_model = net
        self.num_centers = num_centers
        self.optimizer = optimizer
        self.model_from_round = 0
        self.lr = lr
        self.local_epochs = local_epochs
        self.loss = loss
        self.local_val_losses = []
        self.global_val_losses = []
        self.local_train_losses = []
        self.global_train_losses = []
        self.logger = logger
        self.best_model = None
        self.stratify_on = stratify_on
        self.stratify_labels = stratify_labels
        self.set_device(device)
        self.set_members(features, labels, batch_size)

    def device(self):
        return self._device

    def set_device(self, device):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device
        self.global_model.to(self.device())

    def set_members(self, features, labels, batch_size):
        if self.stratify_on == None:
            dict_center_idxs = sample_iid(features, self.num_centers)
        elif self.stratify_labels:
            dict_center_idxs = sample_by_quantiles(labels, self.stratify_on, self.num_centers)
        else:
            dict_center_idxs = sample_by_quantiles(features, self.stratify_on, self.num_centers)
        self.members = []
        for center_idx in range(self.num_centers):
            self.members.append(Member(self.optimizer, self.lr, features, labels, [0.9, 0.1, 0], 
                                    dict_center_idxs[center_idx], self.local_epochs, self.logger, self.loss, batch_size, self._device))

    def get_members(self):
        return self.members

    def fit(self, epochs=1, patience=3, print_every=2, take_best=True, verbose=False):
        self.global_model.train()

        val_loss = []

        epochs_no_improve = 0
        min_val_loss = 100000
        
        for epoch in range(epochs):
            local_weights, local_losses = [], []
            if verbose:
                print(f'\n | Global Training Round : {epoch+1} |\n')

            for member in self.members:
                w, loss = member.update_weights(model=copy.deepcopy(self.global_model), global_round=epoch, verbose=verbose) 
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
            # potentially weight this
            global_weights = average_weights(local_weights)

            #update global model with new weights
            self.global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)

            self.global_train_losses.append(loss_avg)

            # val loss of global model on local data
            local_val_losses = []
            for member in self.members:
                local_val_loss = member.get_loss(copy.deepcopy(self.global_model), validate=True)
                local_val_losses.append(local_val_loss)
            # potentially weight this
            val_loss_avg = sum(local_val_losses) / len(local_val_losses)
            self.global_val_losses.append(val_loss_avg)

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f' \Latest training stats after {epoch+1} global rounds:')
                print(f'Training loss : {self.global_train_losses[-1]}')
                print(f'Validation loss : {val_loss[-1]}') 

            if val_loss_avg < min_val_loss and take_best:
                self.model_from_round = epoch + 1
                self.best_model = global_weights
                min_val_loss = val_loss_avg
                epochs_no_improve = 0
            elif not take_best:
                self.model_from_round = epoch + 1
                self.best_model = global_weights
            else:
                epochs_no_improve += 1
                if epochs_no_improve > patience:
                    print(f'Early stop at epoch {epoch+1}, model from round {self.model_from_round}')
                    torch.save(self.best_model, '.best_model.pt')
                    # get the local losses of each member based on their own model and data
                    for member in self.members:
                        self.local_val_losses.append(member.val_losses)
                        self.local_train_losses.append(member.train_losses)
                    return epoch + 1
            val_loss.append(val_loss_avg)

        print(f'Epochs exhausted, model from round {self.model_from_round}')
        torch.save(self.best_model, '.best_model.pt')
        # get the local losses of each member based on their own model and data
        for member in self.members:
            self.local_val_losses.append(member.val_losses)
            self.local_train_losses.append(member.train_losses)
        return epochs

    def predict_hazard(self, input=None):
        hazard = []
        model = copy.deepcopy(self.global_model)
        model.load_state_dict(self.best_model)
        for member in self.members:
            hazard.append(member.predict_hazard(model, input))
        return hazard

    def predict_surv(self, input=None):
        surv = []
        model = copy.deepcopy(self.global_model)
        model.load_state_dict(self.best_model)
        for member in self.members:
            surv.append(member.predict_surv(model, input))        
        return surv

    def predict_surv_df(self, cuts, input=None):
        surv = []
        model = copy.deepcopy(self.global_model)
        model.load_state_dict(self.best_model)
        for member in self.members:
            surv.append(member.predict_surv_df(model, cuts, input))        
        return surv