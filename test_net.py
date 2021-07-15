import numpy as np
import pytest
import torch

from net import MLP, CoxPH, MLPPH

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