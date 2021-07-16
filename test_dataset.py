import numpy as np
import pandas as pd
import pytest
import torch 

from dataset import sample_iid, sample_benign_malignant

@pytest.mark.parametrize('seed', [0,2,4])
@pytest.mark.parametrize('centers', [2,5,10])
def test_no_overlap_iid(seed,centers):
    N = 100
    indicators = torch.randint(low=0, high=2, size=(N,2)).numpy()
    data = pd.DataFrame(indicators, columns=['SITE_C71', 'SITE_D32'])
    center_idxs = sample_iid(data, centers)
    idxs = []
    for key in center_idxs:
        idxs = idxs + list(center_idxs[key])
    assert(len(idxs) == len(set(idxs))) 



@pytest.mark.parametrize('seed', [0,2,4])
@pytest.mark.parametrize('centers', [2,5,10])
def test_no_overlap_noniid(seed,centers):
    N = 100
    indicators = torch.randint(low=0, high=2, size=(N,6)).numpy()
    data = pd.DataFrame(indicators, columns=['SITE_C71', 'SITE_C70', 'SITE_C72', 'SITE_D32', 'SITE_D33', 'SITE_D35'])
    center_idxs = sample_benign_malignant(data, centers)
    idxs = []
    for key in center_idxs:
        idxs = idxs + list(center_idxs[key])
    assert(len(idxs) == len(set(idxs))) 
