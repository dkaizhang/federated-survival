import numpy as np
import pytest
import torch

from loss import negative_llh
from pycox.models.loss import nll_logistic_hazard

@pytest.mark.parametrize('seed', [0])
def test_negative_llh_matches(seed):
    torch.manual_seed(seed)
    N = 100
    T = 10
    phi = torch.randn((N, T))
    idx_durations = torch.randint(low=0, high=T, size=(N,))
    events = torch.randint(low=0, high=2, size=(N,)).float()
    nll_pycox = nll_logistic_hazard(phi, idx_durations=idx_durations, events=events)    
    nll_mine = negative_llh(phi, idx_durations=idx_durations, events=events)
    assert((nll_pycox - nll_mine).abs() < 1e-5)