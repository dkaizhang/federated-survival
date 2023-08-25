import numpy as np
import torch
import warnings

from Data.data_sim import SimStudyNonLinearNonPH
from Data.data_sim import SimStudyNonLinearNonPHSquared
from Data.data_sim import SimStudyNonLinearNonPHCubed
from Data.data_sim import SimStudyNonLinearNonPHAll
from pycox import datasets
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# TODO refac this out
rng = np.random.default_rng(123)

def load_data(dataset):

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

    # flchain
    # cols_standardise = []
    # cols_minmax = ['age','sample.yr','kappa','lambda','flc.grp','creatinine']
    # cols_leave = ['mgus','sex']

    # simulacrum
    # cols_standardise = ['GRADE', 'AGE', 'QUINTILE_2015', 'NORMALISED_HEIGHT', 'NORMALISED_WEIGHT']
    # cols_minmax = ['SEX', 'TUMOUR_COUNT', 'REGIMEN_COUNT']
    # cols_leave = ['SACT', 'CLINICAL_TRIAL_INDICATOR', 'CHEMO_RADIATION_INDICATOR','BENIGN_BEHAVIOUR','SITE_C70', 'SITE_C71', 'SITE_C72', 'SITE_D32','SITE_D33','SITE_D35','CREG_L0201','CREG_L0301','CREG_L0401','CREG_L0801','CREG_L0901','CREG_L1001','CREG_L1201','CREG_L1701','LAT_9','LAT_B','LAT_L','LAT_M','LAT_R','ETH_A','ETH_B','ETH_C','ETH_M','ETH_O','ETH_U','ETH_W','DAYS_TO_FIRST_SURGERY']

    all_cols = cols_standardise + cols_minmax + cols_leave
    standardise = [(f'standard{i}',StandardScaler(), [col]) for i,col in enumerate(cols_standardise)]
    leave = [(f'leave{i}','passthrough',[col]) for i,col in enumerate(cols_leave)]
    minmax = [(f'minmax{i}',MinMaxScaler(),[col]) for i,col in enumerate(cols_minmax)] 

    x_mapper = ColumnTransformer(standardise + minmax + leave) 

    return all_cols, x_mapper


"""
Argument:
x - DataFrame of features
y - tuple of (durations, events)
x_mapper - ColumnTransformer for all features
discretiser - Discretiser to be applied to y
fit_transform - for x_mapper and discretiser on x and y respectively 
Returns:
x_trans - 
y_trans - tuple of (discretised durations, events)
"""
def data_transform(x, y, x_mapper, discretiser, fit_transform=True):

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

# need to deal with those who have both benign and malignant tumour
# def sample_benign_malignant(data, num_centers):
#     """
#     Split data between benign and malignant tumours then distribute across each centre
#     Arguments:
#     data -- combined data
#     num_centres -- number of centres to spread over    
#     Returns:
#     Dict with centre_id : indices of data assigned to centre
#     """
#     malignant = (data['SITE_C71'] == 1) | (data['SITE_C70'] == 1) | (data['SITE_C72'] == 1)
#     benign = (data['SITE_D32'] == 1) | (data['SITE_D33'] == 1) | (data['SITE_D35'] == 1)    
    
#     malignant = data.loc[malignant]
#     benign = data.loc[benign]

#     malignant_dict = sample_iid(malignant, num_centers // 2)
#     benign_dict = sample_iid(benign, num_centers - num_centers // 2, start_center = num_centers // 2)

#     dict_center_idxs = {}
#     dict_center_idxs.update(malignant_dict)
#     dict_center_idxs.update(benign_dict)

#     return dict_center_idxs    

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