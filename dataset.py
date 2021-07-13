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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.durations = labels[0]
        self.events = labels[1]

    def __getitem__(self, index):
        return self.features[index], self.durations[index], self.events[index]
       
    def __len__(self):
        return len(self.events)