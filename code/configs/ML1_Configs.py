"""Code for setting configs of ML1_Machine_Learning.py"""

from models.gcn import Net as gcn
from models.pna import Net as pna
from models.gnn import Net as gnn
from models.gan import Net as gan
from models.gin import Net as gin

configs = [
    {
        'net': gcn,
        'batch_size': 64,
        'epoch_size': 200,
        'embedding_size': 128,
    },
]
