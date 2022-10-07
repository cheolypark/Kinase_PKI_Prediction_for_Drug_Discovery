"""Code for setting configs of ML2_Machine_Learning_Experiment.py"""

from models.gcn import Net as gcn
from models.pna import Net as pna
from models.gnn import Net as gnn
from models.gan import Net as gan
from models.gin import Net as gin

configs = [
    {
        'net': gin,
        'epoch_size': 200,
        "batch_size": 64,
        "hidden": 512,
        "num_layers": 6,
    },
    {
        'net': gan,
        'epoch_size': 200,
        "batch_size": 64,
        "hidden": 512,
        "num_layers": 6,
    },
    {
        'net': gnn,
        'epoch_size': 200,
        "batch_size": 64,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "sgd_momentum": 0.8,
        "scheduler_gamma": 0.8,
        "pos_weight": 1.3,
        "model_embedding_size": 64,
        "model_attention_heads": 3,
        "model_layers": 4,
        "model_dropout_rate": 0.2,
        "model_top_k_ratio": 0.5,
        "model_top_k_every_n": 1,
        "model_dense_neurons": 256
    },
    {
        'net': pna,
        'batch_size': 64,
        'epoch_size': 200,
        "num_layers": 4,
    },
    {
        'net': gcn,
        'batch_size': 64,
        'epoch_size': 200,
        'embedding_size': 128,
    },
]
