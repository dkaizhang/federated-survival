import numpy as np
from numpy.random import sample
import pandas as pd
import pytest
import torch 


from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from dataset import sample_iid, sample_by_quantiles
from load import read_csv


@pytest.mark.parametrize('seed', [0,2,4])
@pytest.mark.parametrize('centers', [2,5,10])
def test_no_overlap_iid(seed,centers):
    N = 100
    data = torch.randint(low=0, high=2, size=(N,6)).numpy()
    # data = pd.DataFrame(indicators, columns=['SITE_C71', 'SITE_C70', 'SITE_C72', 'SITE_D32', 'SITE_D33', 'SITE_D35']).to_numpy()
    center_idxs = sample_iid(data, centers)
    idxs = []
    for key in center_idxs:
        idxs = idxs + list(center_idxs[key])
    assert(len(center_idxs.keys()) == centers)
    assert(len(idxs) == len(set(idxs))) 


@pytest.mark.parametrize('seed', [0,2,4])
@pytest.mark.parametrize('centers', [2,5,10])
@pytest.mark.parametrize('start_center', [0,1,2])
def test_keys_iid(seed,centers, start_center):
    N = 100
    data = torch.randint(low=0, high=2, size=(N,6)).numpy()
    # data = pd.DataFrame(indicators, columns=['SITE_C71', 'SITE_C70', 'SITE_C72', 'SITE_D32', 'SITE_D33', 'SITE_D35'])
    center_idxs = sample_iid(data, centers, start_center=start_center)
    for i, k in enumerate(center_idxs.keys()):
        assert(int(k) == i + start_center)


@pytest.mark.parametrize('seed', [0,2,4])
@pytest.mark.parametrize('centers', [2,5,10])
def test_keys_quantile(seed,centers):
    N = 100
    data = torch.rand(size=(N,3)).numpy()
    age_col = 1
    center_idxs = sample_by_quantiles(data, age_col, centers)
    all_centers = [i for i in range(centers)]
    for i, k in enumerate(center_idxs.keys()):
        assert(int(k) == all_centers[i])
    assert(len(center_idxs.keys()) == centers)

@pytest.mark.parametrize('seed', [0,2,4])
@pytest.mark.parametrize('centers', [2,5,10])
def test_no_overlap_quantile(seed,centers):
    N = 100
    data = torch.rand(size=(N,3)).numpy()
    age_col = 1
    center_idxs = sample_by_quantiles(data, age_col, centers)

    idxs = []
    prev_max = 0
    for key in center_idxs:
        idxs = idxs + list(center_idxs[key])
        m = data.T[age_col][center_idxs[key]].max()
        assert(m > prev_max)
        prev_max = m
    assert(len(idxs) == len(set(idxs))) 
    assert(len(idxs) == N) 
    

@pytest.mark.parametrize('seed', [0,2,4])
@pytest.mark.parametrize('centers', [2,5,10])
def test_no_overlap_quantile(seed,centers):
    N = 100
    datapath = './Data/data.csv'
    data = read_csv(datapath).head(N)

    cols_standardise = ['GRADE', 'QUINTILE_2015', 'NORMALISED_HEIGHT', 'NORMALISED_WEIGHT']
    cols_minmax = ['SEX', 'TUMOUR_COUNT', 'REGIMEN_COUNT']
    cols_leave = ['AGE','SACT', 'CLINICAL_TRIAL_INDICATOR', 'CHEMO_RADIATION_INDICATOR','BENIGN_BEHAVIOUR','SITE_C70', 'SITE_C71', 'SITE_C72', 'SITE_D32','SITE_D33','SITE_D35','CREG_L0201','CREG_L0301','CREG_L0401','CREG_L0801','CREG_L0901','CREG_L1001','CREG_L1201','CREG_L1701','LAT_9','LAT_B','LAT_L','LAT_M','LAT_R','ETH_A','ETH_B','ETH_C','ETH_M','ETH_O','ETH_U','ETH_W','DAYS_TO_FIRST_SURGERY']

    all_cols = cols_standardise + cols_minmax + cols_leave

    standardise = [([col], StandardScaler()) for col in cols_standardise]
    minmax = [([col], MinMaxScaler()) for col in cols_minmax]
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(standardise + minmax + leave)

    x = x_mapper.fit_transform(data).astype('float32')

    stratify_col = 'AGE'
    stratify_on = all_cols.index(stratify_col)

    print(x.T[stratify_on])

    center_idxs = sample_by_quantiles(x, stratify_on, centers)

    idxs = []
    prev_max = -99999
    for key in center_idxs:
        idxs = idxs + list(center_idxs[key])
        print(key)
        print(center_idxs[key])
        print(x.T[stratify_on][center_idxs[key]])
        m = x.T[stratify_on][center_idxs[key]].max()
        assert(m > prev_max)
        prev_max = m
    assert(len(idxs) == len(set(idxs))) 
    assert(len(idxs) == N) 






# @pytest.mark.parametrize('seed', [0,2,4])
# @pytest.mark.parametrize('centers', [2,5,10])
# def test_keys_benign_malignant(seed,centers):
#     N = 100
#     indicators = torch.randint(low=0, high=2, size=(N,6)).numpy()
#     data = pd.DataFrame(indicators, columns=['SITE_C71', 'SITE_C70', 'SITE_C72', 'SITE_D32', 'SITE_D33', 'SITE_D35'])
#     center_idxs = sample_benign_malignant(data, centers)
#     all_centers = [i for i in range(centers)]
#     for i, k in enumerate(center_idxs.keys()):
#         assert(int(k) == all_centers[i])

# overlap as some have both
# @pytest.mark.parametrize('seed', [0,2,4])
# @pytest.mark.parametrize('centers', [2,5,10])
# def test_no_overlap_benign_malignant(seed,centers):
#     N = 100
#     indicators = torch.randint(low=0, high=2, size=(N,6)).numpy()
#     data = pd.DataFrame(indicators, columns=['SITE_C71', 'SITE_C70', 'SITE_C72', 'SITE_D32', 'SITE_D33', 'SITE_D35'])
#     center_idxs = sample_benign_malignant(data, centers)
#     idxs = []
#     for key in center_idxs:
#         idxs = idxs + list(center_idxs[key])
#     assert(len(center_idxs.keys()) == centers)
#     assert(len(idxs) == len(set(idxs))) 
