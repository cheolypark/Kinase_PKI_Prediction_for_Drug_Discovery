"""GIN"""

#===================================================================================================================
# Code reference:
#  I referred to the code in the below link.
#    https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gin.py
#
#===================================================================================================================


import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential

from torch_geometric.nn import GINConv, global_mean_pool

torch.manual_seed(42)


class Net(torch.nn.Module):
    name = 'GIN'

    def __init__(self, dataset, cf):
        super().__init__()

        num_features = dataset.num_features
        hidden = cf['hidden']
        num_layers = cf['num_layers']

        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=True)

        self.convs = torch.nn.ModuleList()

        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=True))

        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 1)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch_index = data.batch

        x = self.conv1(x, edge_index)

        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_mean_pool(x, batch_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x

    def change_data_to_address_this_model(self, data):
        pass
