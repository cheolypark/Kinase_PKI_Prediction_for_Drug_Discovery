"""Code for supporting main codes"""

from matplotlib import pyplot as plt
import os.path as osp
import torch


def save_image(dir, name):
    plt.xticks(rotation=45)
    plt.savefig(osp.join(dir, name + ".png"), dpi=600)
    plt.close()


def print_data_info(data):
    print("Dataset type: ", type(data))
    print("Dataset features: ", data.num_features)
    print("Dataset length: ", data.len)
    print("Dataset sample: ", data[0])
    print("Sample  nodes: ", data[0].num_nodes)
    print("Sample  edges: ", data[0].num_edges)


def print_gpu_info(device):
    print('Using device:', device)

    if str(device) == 'cuda':
        print('Device Count:', torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


def convert_to_one_value(lst):
    ret = 0
    for i, v in enumerate(lst):
        new_v = v*(10**i)
        ret += new_v

    return ret


def convert_tensor_to_one_value(tsr, selected_index=[0, 1]):
    ret = []
    for t in tsr:
        lst = [x for i, x in enumerate(t.tolist()) if i in selected_index]
        new_v = convert_to_one_value(lst)
        ret.append([new_v])

    return torch.tensor(ret)
