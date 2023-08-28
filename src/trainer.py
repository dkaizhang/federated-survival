import copy
import os
import torch

from src.loss import negative_llh
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# this should own the fed centers
class Trainer():
    def __init__(self, num_centers, epochs, local_epochs, batch_size, optimizer, lr, num_workers, device, seed, writer):
        
        self.num_centers = num_centers
        self.epochs = epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.seed = seed
        self.writer = writer

    def average_weights(self, w):
        """
        Arguments:
        w -- list of model state dicts.
        Returns:
        w_avg -- the average of the weights.
        """
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg

    def fit(self, model, train, dict_center_idxs, val, val_dict_center_idxs):

        # create federation members - maybe replace federation class with this trainer
        # trainer should act as modelwrapper with all the training args and directly implement trianing loop
        # might not need to save local models as we use global aggregation

        center_data = [Subset(train, list(dict_center_idxs[k])) for k in dict_center_idxs.keys()]
        center_step = [0 for i in range(len(center_data))]

        center_val_data = [Subset(val, list(val_dict_center_idxs[k])) for k in val_dict_center_idxs.keys()]

        for g_epoch in range(self.epochs):

            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            model_dicts = []
            for c_id, c_data in enumerate(center_data):

                print(f"---Center {1+c_id} / {len(center_data)}")
                local_model = copy.deepcopy(model)
                center = Center(model=local_model, optimizer=self.optimizer, lr=self.lr, device=self.device)

                for l_epoch in tqdm(range(self.local_epochs)):
                    train_loader = DataLoader(c_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
                    for data in train_loader:
                        batch_c_loss = center.local_step(data)
                        self.writer.add_scalar(f"Loss {c_id} - train", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    c_val_loss = 0
                    val_loader = DataLoader(center_val_data[c_id], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                    for data in val_loader:
                        c_val_loss += center.local_test_step(data)
                    c_val_loss = c_val_loss / len(val_loader)
                    self.writer.add_scalar(f"Loss {c_id} - val", c_val_loss, center_step[c_id])

                model_dicts.append(copy.deepcopy(center.get_model_dict())) 

                del center

            weights = self.average_weights(model_dicts)
            model.load_state_dict(weights)

        save_to = os.path.join(self.writer.log_dir, f"model-{g_epoch}.pt")
        torch.save(model.state_dict(), save_to)

class Center():
    def __init__(self, model, optimizer, lr, device) -> None:

        self.model = model
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            print('optimizer not implemented')
            exit(0)
        self.device = device

    def local_step(self, data):
        self.model.to(self.device)
        self.model.train()

        x, durations, events = data
        x = x.to(self.device)
        durations = durations.to(self.device)
        events = events.to(self.device)

        self.optimizer.zero_grad()
        phi = self.model(x)
        loss = negative_llh(phi, durations, events)
        loss.backward()
        self.optimizer.step()

        return loss
    
    def get_model_dict(self):
        return self.model.state_dict()

    def local_test_step(self, data):    
        self.model.to(self.device)
        self.model.eval()

        x, durations, events = data
        x = x.to(self.device)
        durations = durations.to(self.device)
        events = events.to(self.device)

        with torch.no_grad():
            phi = self.model(x)
            loss = negative_llh(phi, durations, events)
        
        self.model.train()
        return loss

'''
class Federation():
    """
    Accepts data, creates Members and distributes data
    Accepts a neural net and performs the training with the nll as loss function (set in Member)
    """
    def __init__(self, model, num_centers, optimizer, lr,  batch_size=256, local_epochs=1, loss=negative_llh,  device=None):

        self.global_model = model
        self.num_centers = num_centers
        self.optimizer = optimizer
        self.lr = lr


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
            # using raw labels instead of discretised labels
            dict_center_idxs = sample_by_quantiles(self.raw_labels, self.stratify_on, self.num_centers)
            # dict_center_idxs = sample_by_quantiles(labels, self.stratify_on, self.num_centers)
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
'''