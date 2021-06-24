import numpy as np
import torch

rng = np.random.default_rng(123)


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

class Discretiser():
    def __init__(self, cuts):
        self._cuts = cuts
        self.cuts = None

    def fit(self, durations):
        durations = durations.astype(np.float64)
        self.cuts = np.linspace(0, durations.max(), self._cuts, dtype=np.float64)
        return self
    
    def transform(self, durations, events):    
        idx_durations = np.digitize(durations, self.cuts) - 1 + events
        idx_durations = idx_durations.clip(0, len(self.cuts) - 1)
        # idx_durations = np.digitize(durations, np.concatenate((self.cuts, [np.infty]), dtype=np.float64)) - 1 + events
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