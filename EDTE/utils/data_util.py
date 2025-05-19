import os
from symbol import shift_expr
import numpy as np
import torch
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
import pickle
from .mutils import seed_everything
from pathlib import Path

root = Path(__file__).parent.parent.parent


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def select_by_field(edges, fields=[0, 1]):
    # field [0,1,2,3,4]
    res = []
    for f in fields:
        e = edges[edges[:, 4] == f]
        res.append(e)
    edges = torch.cat(res, dim=0)
    res = []
    for i in range(16):
        e = edges[edges[:, 2] == i]
        e = e[:, :2]
        res.append(e)
    edges = res
    return edges


def select_by_venue(edges, venues=[0, 1]):
    # venue [0-21]
    res = []
    for f in venues:
        e = edges[edges[:, 3] == f]
        res.append(e)
    edges = torch.cat(res, dim=0)
    res = []
    for i in range(16):
        e = edges[edges[:, 2] == i]
        e = e[:, :2]
        res.append(e)
    edges = res
    return edges


def load_data(args):
    seed_everything(0)
    dataset = args.dataset
    data = torch.load(root / 'data' / dataset)
    if dataset.startswith("collab"):
        from ..data_configs.collab import (
            testlength,
            vallength,
            length,
        )

        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        args.length = length
        args.nfeat = data["x"].shape[1]
        args.num_nodes = len(data["x"])
    elif dataset.startswith("epinions"):
        from ..data_configs.epinions import testlength, vallength, length, num_nodes

        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        args.length = length
        args.nfeat = data["x"].shape[1]
        args.num_nodes = num_nodes

    elif dataset.startswith("bitcoinalpha"):
        from ..data_configs.bitcoinalpha import testlength, vallength, length, num_nodes

        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        args.length = length
        args.nfeat = data["x"].shape[1]
        args.num_nodes = num_nodes

    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")
    print(f"Loading dataset {dataset}")
    return args, data
