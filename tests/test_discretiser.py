import numpy as np
import pytest

from model.discretiser import Discretiser
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

@pytest.mark.parametrize('seed', [0,2,4])
@pytest.mark.parametrize('cuts', [3,5,10])
def test_equidistant_cuts_match(seed,cuts):
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


@pytest.mark.parametrize('seed', [0,2,4])
@pytest.mark.parametrize('cuts', [3,5,10])
def test_km_cuts_match(seed,cuts):
    rng = np.random.default_rng(seed)

    N = 2000
    durations = rng.integers(low=0, high=10000, size=N)
    events = rng.integers(low=0,high=2,size=N)
    labtrans = LabTransDiscreteTime(cuts, scheme='quantiles')
    idx_d_lab, events_lab = labtrans.fit_transform(durations, events)

    discretiser = Discretiser(cuts, scheme='km')
    idx_d_dis, events_dis = discretiser.fit_transform(durations, events)

    assert((idx_d_lab == idx_d_dis - 1) | (idx_d_lab == idx_d_dis) | (idx_d_lab == idx_d_dis + 1)).all()
    assert(events_lab == events_dis).all()
    print(seed, cuts)
    print(discretiser.cuts)
    print(labtrans.cuts)
    assert((discretiser.cuts >= labtrans.cuts - 1) | (discretiser.cuts <= labtrans.cuts + 1)).all()