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


def sample_benign_malignant(data, num_centers):
    """
    Split data between benign and malignant tumours then distribute across each centre
    Arguments:
    data -- combined data
    num_centres -- number of centres to spread over    
    Returns:
    Dict with centre_id : indices of data assigned to centre
    """
    malignant = (data['SITE_C71'] == 1) | (data['SITE_C70'] == 1) | (data['SITE_C72'] == 1)
    benign = (data['SITE_D32'] == 1) | (data['SITE_D33'] == 1) | (data['SITE_D35'] == 1)    
    
    malignant = data.loc[malignant]
    benign = data.loc[benign]

    malignant_dict = sample_iid(malignant, num_centers // 2)
    benign_dict = sample_iid(benign, num_centers - num_centers // 2)

    dict_center_idxs = {}
    dict_center_idxs.update(malignant_dict)
    dict_center_idxs.update(benign_dict)

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