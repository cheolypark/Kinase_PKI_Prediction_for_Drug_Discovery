"""PNA Architecture"""

#===================================================================================================================
# Code reference:
#  I referred to the code in the below link.
#    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pna.py
#
#===================================================================================================================


import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
from torch_geometric.utils import degree
from util.helper import convert_tensor_to_one_value


def get_maximum_in_degree(dataset):
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg


class Net(torch.nn.Module):
    name = 'PNA'

    def __init__(self, dataset, cf):
        super().__init__()

        deg = get_maximum_in_degree(dataset)
        num_layers = cf['num_layers']

        # Original Setting
        # self.node_emb = Embedding(21, 75)
        # self.edge_emb = Embedding(4, 50)
        # Single value case
        # self.node_emb = Embedding(65, 75)
        # self.edge_emb = Embedding(65, 50)
        # Multiple value case
        self.node_emb = Embedding(10000, 75)
        self.edge_emb = Embedding(100, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(num_layers):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, 1))

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch_index = data.batch

        x = torch.tensor(x).to(torch.int64)
        edge_attr = torch.tensor(edge_attr).to(torch.int64)

        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch_index)
        return self.mlp(x)

    def change_data_to_address_this_model(self, data):
        # Multi-values case
        atomic_num = 0
        atomic_degree = 1
        measurement_type = 15
        kinase_type = 16

        data.x = convert_tensor_to_one_value(data.x, [atomic_num, atomic_degree, measurement_type, kinase_type])
        data.edge_attr = convert_tensor_to_one_value(data.edge_attr).squeeze()

        # Simple approach using only atom number
        # t = torch.tensor([0])
        # data.x = torch.index_select(data.x, 1, t)
        # data.edge_attr = torch.index_select(data.edge_attr, 1, t)
        # data.edge_attr = torch.reshape(data.edge_attr, (-1,))
