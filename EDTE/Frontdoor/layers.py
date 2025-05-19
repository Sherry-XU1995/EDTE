from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor

from .cell import *


## spatial
class HodgeLaguerreConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None):
        norm = edge_weight
        Tx_0 = x
        Tx_1 = x
        out = self.lins[0](Tx_0)
        xshape = x.shape
        k = 1

        if len(self.lins) > 1:
            x = x.reshape(xshape[0], -1)
            Tx_1 = x - self.propagate(edge_index, x=x, norm=norm, size=None)
            if len(xshape) >= 3:
                Tx_1 = Tx_1.view(xshape[0], xshape[1], -1)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            inshape = Tx_1.shape
            Tx_1 = Tx_1.view(inshape[0], -1)
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            if len(xshape) >= 3:
                Tx_2 = Tx_2.view(inshape[0], inshape[1], -1)
                Tx_1 = Tx_1.view(xshape[0], xshape[1], -1)
            Tx_2 = (-Tx_2 + (2 * k + 1) * Tx_1 - k * Tx_0) / (k + 1)
            k += 1
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}')


class GCNConv(MessagePassing):
    def __init__(self, in_channels, num_nodes, out_channels, K):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.layer_norm = nn.LayerNorm([num_nodes, out_channels])
        self.bias = Parameter(torch.Tensor(out_channels))
        self.K = K
        self.num_nodes = num_nodes
        self.out_channels = out_channels

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, input, edge_index, edge_weight):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        x = self.lin(input)
        x_res = x
        edge_index = self.undir2dir(edge_index)
        edge_weight = edge_weight.reshape(-1, self.K)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=edge_weight[..., k])
            x = F.relu(x)

        out = self.bias + x_res + x
        out = self.layer_norm(out.reshape(-1, self.num_nodes, self.out_channels))
        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        return norm.view(-1, 1) * x_j

    def undir2dir(self, edge_index):
        src, dst = edge_index[0], edge_index[1]
        directed_edge_index = torch.stack([src, dst], dim=0)
        reversed_edge_index = torch.stack([dst, src], dim=0)
        edge_index = torch.cat([directed_edge_index, reversed_edge_index], dim=1)
        return edge_index
