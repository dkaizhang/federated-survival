import torch 

from torch import nn

def load_model(model_type, dim_in, dim_out, model_path=None):
    
    num_nodes = [32, 32]
    if model_type == 'NNnph':   
        net = MLP(dim_in=dim_in, num_nodes=num_nodes, dim_out=dim_out)    
    if model_type == 'CoxPH':            
        net = CoxPH(dim_in=dim_in, dim_out=dim_out)
    if model_type == 'NNph':
        net = MLPPH(dim_in=dim_in, num_nodes=num_nodes, dim_out=dim_out)
    else:
        ValueError

    if model_path is not None:
        net.load_state_dict(torch.load(model_path))


class DenseBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, batch_norm=True, dropout=0, activation=nn.ReLU, 
                    w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias)
        # fill in self.linear.weight.data with kaiming normal
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(dim_out) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input

# output is unconstrained if output_activation not sigmoid
class MLP(nn.Module):
    def __init__(self, dim_in, num_nodes, dim_out, batch_norm=True, dropout=None, activation=nn.ReLU,
                output_activation=None, output_bias=True,
                w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        num_nodes = [dim_in] + num_nodes
        net = []
        for d_in, d_out in zip(num_nodes[:-1], num_nodes[1:]):
            net.append(DenseBlock(d_in, d_out, bias=True, batch_norm=batch_norm, 
                        dropout=dropout, activation=activation, w_init_=w_init_))
        # print(num_nodes[-1], dim_out, output_bias)
        # print(type(num_nodes[-1]), type(dim_out))
        net.append(nn.Linear(num_nodes[-1], dim_out, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)

class PHBlock(nn.Module):
    def __init__(self, dim_in, dim_out, batch_norm=True):
        super().__init__()
        self.dim_out = dim_out
        self.linear1 = nn.Linear(dim_in, 1, bias=False)
        self.batch_norm = nn.BatchNorm1d(1) if batch_norm else None
        self.linear2 = nn.Linear(1, dim_out, bias=True)
        torch.nn.init.ones_(self.linear2.weight.data)
        torch.nn.init.normal_(self.linear2.bias.data, mean=0.0, std=0.5)
        self.linear2.weight.requires_grad = False


    def forward(self, input):
        input = self.linear1(input)
        if self.batch_norm:
            input = self.batch_norm(input)
        input = self.linear2(input)
        return input
    

class CoxPH(nn.Module):
    def __init__(self, dim_in, dim_out, batch_norm=True, output_activation=None):
        super().__init__()
        net = []
        net.append(PHBlock(dim_in, dim_out, batch_norm))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)

class MLPPH(nn.Module):
    def __init__(self, dim_in, num_nodes, dim_out, batch_norm=True, dropout=None, activation=nn.ReLU,
                output_activation=None, output_bias=True,
                w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        num_nodes = [dim_in] + num_nodes
        net = []
        for d_in, d_out in zip(num_nodes[:-1], num_nodes[1:]):
            net.append(DenseBlock(d_in, d_out, bias=True, batch_norm=batch_norm, 
                        dropout=dropout, activation=activation, w_init_=w_init_))
        # print(num_nodes[-1], dim_out, output_bias)
        # print(type(num_nodes[-1]), type(dim_out))
        net.append(PHBlock(num_nodes[-1], dim_out, batch_norm))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)