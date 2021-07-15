import numpy as np
import pytest
import torch

from copy import deepcopy
from loss import negative_llh
from net import MLP, CoxPH, MLPPH, PHBlock

@pytest.mark.parametrize('dim_out', [1,5,15])
@pytest.mark.parametrize('seed', [0,2,4])
def test_net_dimensions(dim_out, seed):
    torch.manual_seed(seed)
    N = 5
    dim_in = 10 
    num_nodes = [32, 32]
    net = MLP(dim_in=dim_in, num_nodes=num_nodes, dim_out=dim_out)

    input = torch.randn((N, dim_in))
    output = net(input)
    assert(output.shape[0] == N and output.shape[1] == dim_out)

@pytest.mark.parametrize('dim_out', [1,5,15])
@pytest.mark.parametrize('seed', [0,2,4])
def test_CoxPH_dimensions(dim_out, seed):
    torch.manual_seed(seed)
    N = 5
    dim_in = 10 
    net = CoxPH(dim_in=dim_in, dim_out=dim_out)

    input = torch.randn((N, dim_in))
    output = net(input)
    print(output)
    assert(output.shape[0] == N and output.shape[1] == dim_out)

@pytest.mark.parametrize('dim_out', [1,5,15])
@pytest.mark.parametrize('seed', [0,2,4])
def test_MLPPH_dimensions(dim_out, seed):
    torch.manual_seed(seed)
    N = 5
    dim_in = 10 
    num_nodes = [32, 32]
    net = MLPPH(dim_in=dim_in, num_nodes=num_nodes, dim_out=dim_out)

    input = torch.randn((N, dim_in))
    output = net(input)
    assert(output.shape[0] == N and output.shape[1] == dim_out)

@pytest.mark.parametrize('dim_out', [1,5,15])
@pytest.mark.parametrize('seed', [0,2,4])
def test_PHBlock_weights(dim_out, seed):
    torch.manual_seed(seed)
    N = 5

    dim_in = 10 
    model = PHBlock(dim_in,dim_out)

    linear1w = deepcopy(model.linear1.weight.data)
    linear2w = deepcopy(model.linear2.weight.data)
    linear2b = deepcopy(model.linear2.bias.data)

    model.zero_grad()
    input = torch.randn((N, dim_in))
    idx_durations = torch.randint(low=0, high=dim_out, size=(N,))
    events = torch.randint(low=0, high=2, size=(N,)).float()
    phi = model(input)
    loss = negative_llh(phi, idx_durations, events)
    loss.backward()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
    optimizer.step()

    assert((linear1w != model.linear1.weight.data).any())
    assert((linear2w == model.linear2.weight.data).all())
    assert((linear2b != model.linear2.bias.data).any())

