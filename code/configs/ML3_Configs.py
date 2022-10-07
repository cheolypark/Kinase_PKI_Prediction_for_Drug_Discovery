"""Code for setting configs of ML3_Machine_Learning_Epoch_Search.py"""

from models.gcn import Net as gcn
from models.pna import Net as pna
from models.gnn import Net as gnn
from models.gan import Net as gan
from models.gin import Net as gin

configs = [
    {
        'net': pna,
        'batch_size': 64,
        'epoch_size': 200,
        "num_layers": 24,
    },
]
