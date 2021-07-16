import numpy as np
from numpy.random import sample
import pandas as pd
import pytest
import torch 

from dataset import sample_iid, sample_by_quantiles

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
