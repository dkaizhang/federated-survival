from torch import nn

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