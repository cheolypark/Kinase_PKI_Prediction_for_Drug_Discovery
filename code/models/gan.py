"""GlobalAttentionNet GAN"""

#===================================================================================================================
# Code reference:
#  I referred to the code in the below link.
#    https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/global_attention.py
#
#===================================================================================================================

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import AttentionalAggregation, SAGEConv

torch.manual_seed(42)


class Net(torch.nn.Module):
    name = 'GAN'

    def __init__(self, dataset, cf):
        super().__init__()

        num_features = dataset.num_features
        hidden = cf['hidden']
        num_layers = cf['num_layers']

        self.conv1 = SAGEConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))

        self.att = AttentionalAggregation(Linear(hidden, 1))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 1)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch_index = data.batch

        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.att(x, batch_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def change_data_to_address_this_model(self, data):
        pass
