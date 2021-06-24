import numpy as np
import pytest
import torch

from dataset import Discretiser
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

@pytest.mark.parametrize('seed', [0,2,4])
@pytest.mark.parametrize('cuts', [3,5,10])
def test_cuts_match(seed,cuts):
    rng = np.random.default_rng(seed)

    N = 5
    durations = rng.integers(low=0, high=10000, size=N)
    events = rng.integers(low=0,high=2,size=N)
    labtrans = LabTransDiscreteTime(cuts)
    idx_d_lab, events_lab = labtrans.fit_transform(durations, events)

    discretiser = Discretiser(cuts)
    idx_d_dis, events_dis = discretiser.fit_transform(durations, events)

    assert(idx_d_lab == idx_d_dis).all()
    assert(events_lab == events_dis).all()
    assert(discretiser.cuts == labtrans.cuts).all()
