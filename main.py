
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim


import argparse
import time
import fnmatch
import os.path as osp

from model.net import DGCNet


def find_checkpoint(path, pattern='*.pth'):
    for _, _, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return name
    return None


# Argument parsing
parser = argparse.ArgumentParser(description='DGCNet')

parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#for epoch in range(args.n_epoch):
