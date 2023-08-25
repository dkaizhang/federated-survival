import numpy as np
import torch
import warnings

from Data.data_sim import SimStudyNonLinearNonPH
from Data.data_sim import SimStudyNonLinearNonPHSquared
from Data.data_sim import SimStudyNonLinearNonPHCubed
from Data.data_sim import SimStudyNonLinearNonPHAll
from pycox import datasets
from src.discretiser import Discretiser
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# TODO refac this out
rng = np.random.default_rng(123)

class TabularSurvivalDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.durations = labels[0]
        self.events = labels[1]

    def __getitem__(self, index):
        return self.features[index], self.durations[index], self.events[index]
       
    def __len__(self):
        return len(self.events)

# splits DataFrame into two
def train_test_split(data, test_split, seed):

    test = data.sample(frac=test_split, random_state=seed)
    train = data.drop(test.index)

    return train, test

def load_data(dataset, num_durations, seed):

    if dataset == 'metabric':
        data = datasets.metabric.read_df()
    elif dataset == 'support':
        data = datasets.support.read_df()
    elif dataset == 'gbsg':
        data = datasets.gbsg.read_df()
    elif dataset == 'rr_nl_nhp':
        data = datasets.rr_nl_nhp.read_df()
    elif dataset == 'simulated':
        n = 4000
        sims = [SimStudyNonLinearNonPH(), SimStudyNonLinearNonPHSquared(), SimStudyNonLinearNonPHCubed(), SimStudyNonLinearNonPHAll()]
        sim = sims[0]
        data = sim.simulate(n)
        data = sim.dict2df(data, True)
        data = data.drop(columns=['duration_true','event_true','censoring_true']) 

    data = data.astype({'event' : int})
    train_data, test_data = train_test_split(data, test_split=0.1, seed=seed)
    train_data, val_data = train_test_split(train_data, test_split=0.1, seed=seed)

    all_cols, x_mapper = get_standardiser(dataset)
    discretiser = Discretiser(num_durations, scheme='km') 

    train_data_trans = data_transform(train_data, all_cols, x_mapper, discretiser, fit_transform=True)
    val_data_trans = data_transform(val_data, all_cols, x_mapper, discretiser, fit_transform=False)
    test_data_trans = data_transform(test_data, all_cols, x_mapper, discretiser, fit_transform=False)

    # using transformed labels for training only
    train = TabularSurvivalDataset(features=train_data_trans[all_cols], labels=(train_data_trans.duration.values, train_data_trans.event.values))
    val = TabularSurvivalDataset(features=val_data_trans[all_cols], labels=(val_data.duration.values, val_data.event.values))
    test = TabularSurvivalDataset(features=test_data_trans[all_cols], labels=(test_data.duration.values, test_data.event.values))

    return train, val, test

def get_standardiser(dataset):

    if dataset == 'metabric':
        cols_standardise = []
        cols_minmax = ['x0', 'x1', 'x2', 'x3','x8']
        cols_leave = ['x4','x5','x6','x7']
    elif dataset == 'support':
        cols_standardise = []
        cols_minmax = ['x0','x2','x3','x6','x7', 'x8', 'x9','x10','x11','x12','x13']
        cols_leave = ['x1','x4','x5']
    elif dataset == 'gbsg':
        cols_standardise = []
        cols_minmax = ['x3', 'x4','x5', 'x6']
        cols_leave = ['x0','x1','x2']
    elif dataset == 'simulated':
        cols_standardise = []
        cols_minmax = ['x0', 'x1', 'x2']
        cols_leave = []

    all_cols = cols_standardise + cols_minmax + cols_leave
    standardise = [(f'standard{i}',StandardScaler(), [col]) for i,col in enumerate(cols_standardise)]
    leave = [(f'leave{i}','passthrough',[col]) for i,col in enumerate(cols_leave)]
    minmax = [(f'minmax{i}',MinMaxScaler(),[col]) for i,col in enumerate(cols_minmax)] 

    x_mapper = ColumnTransformer(standardise + minmax + leave) 

    return all_cols, x_mapper


"""
Argument:
data - DataFrame of features and labels
all_cols - List of all features
x_mapper - ColumnTransformer for all features
discretiser - Discretiser to be applied to y
fit_transform - for x_mapper and discretiser on x and y respectively 
Returns:
x_trans - 
y_trans - tuple of (discretised durations, events)
"""
def data_transform(data, all_cols, x_mapper, discretiser, fit_transform=True):

    x = data[all_cols]
    y = (data.duration.values, data.event.values)

    if fit_transform:
        x_trans = x_mapper.fit_transform(x).astype('float32')
    else:
        x_trans = x_mapper.transform(x).astype('float32')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fit_transform:
            y_trans = discretiser.fit_transform(*y)
        else:
            y_trans = discretiser.transform(*y)   

    return x_trans, y_trans

"""
Argument:
df - simple DataFrame with all features and labels 
t_index - rows to be assigned to train
v_index - rows to be assigned to val
features_headers - list of feature names
Returns:
x_train, x_val - df projection containing only features 
y_train, y_val- tuple of (durations, events)
"""
def train_val_split(df, t_index, v_index, feature_headers):
    df_train = df.loc[t_index]
    df_val = df.loc[v_index]

    x_train = df_train[feature_headers]
    y_train = (df_train.duration.values, df_train.event.values)
    x_val = df_val[feature_headers]
    y_val = (df_val.duration.values, df_val.event.values)

    return x_train, y_train, x_val, y_val


def sample_iid(data, num_centers, start_center = 0):
    """
    Randomly split data evenly across each centre
    Arguments:
    data -- combined data
    num_centres -- number of centres to spread over    
    Returns:
    Dict with centre_id : indices of data assigned to centre
    """
    num_items = int(len(data)/num_centers)
    # all_idxs doesn't look at the actual existing indices in data
    dict_center_idxs, all_idxs = {}, [i for i in range(len(data))]
    for i in range(num_centers):
        i = i + start_center

        dict_center_idxs[i] = set(rng.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_center_idxs[i])
    return dict_center_idxs    

def sample_by_quantiles(data, column, num_centers):
    """
    Randomly split data by age groups
    Arguments:
    data -- combined data as numpy array
    column -- column index on which to stratify
    num_centres -- number of centres to spread over    
    Returns:
    Dict with centre_id : indices of data assigned to centre
    """
    
    # labels are tuples, features are DFs
    if type(data) is tuple: 
        data = data[column]
    else:
        data = data.T[column]
    dict_center_idxs, all_idxs = {}, np.array([i for i in range(len(data))])
    quantile = 1 / num_centers
    previous_idxs = torch.zeros(len(data),dtype=torch.bool).numpy()

    print('Available: ',len(data))

    for i in range(num_centers):
        if quantile > 1:
            ValueError
        cutoff = np.quantile(data,quantile)
        selected_idxs = data <= cutoff 
        idxs_in_quantile = selected_idxs & ~previous_idxs
        previous_idxs = previous_idxs | idxs_in_quantile
        dict_center_idxs[i] = all_idxs[idxs_in_quantile]
        quantile += 1 / num_centers 
        # print(np.sum(idxs_in_quantile))

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

