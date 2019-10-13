import numpy as np
import os
from os import path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import argparse
import time
import fnmatch

from model.net import DGCNet
from data.dataset import HPatchesDataset
from utils.evaluate import calculate_epe_hpatches, calculate_pck_hpatches


# Argument parsing
parser = argparse.ArgumentParser(description='DGC-Net')
# Paths
parser.add_argument('--csv-path', type=str, default='data/csv',
                    help='path to training transformation csv folder')
parser.add_argument('--image-data-path', type=str,
                    default='data/hpatches-geometry',
                    help='path to folder containing training images')
parser.add_argument('--model', type=str, default='dgc',
                    help='Model to use', choices=['dgc', 'dgcm'])
parser.add_argument('--metric', type=str, default='aepe',
                    help='Model to use', choices=['aepe', 'pck'])
parser.add_argument('--batch-size', type=int, default=1,
                    help='evaluation batch size')
parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Image normalisation
mean_vector = np.array([0.485, 0.456, 0.406])
std_vector = np.array([0.229, 0.224, 0.225])
normTransform = transforms.Normalize(mean_vector, std_vector)
dataset_transforms = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

# Model
checkpoint_fname = osp.join('pretrained_models', args.model, 'checkpoint.pth')
if not osp.isfile(checkpoint_fname):
    raise ValueError('check the snapshots path')

if args.model == 'dgc':
    net = DGCNet()
elif args.model == 'dgcm':
    net = DGCNet(mask=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net.load_state_dict(torch.load(checkpoint_fname, map_location=device)['state_dict'])
net = nn.DataParallel(net)
net.eval()
net = net.to(device)

with torch.no_grad():
    number_of_scenes = 5
    if (args.metric == 'aepe'):
        res = []
        jac = []
    if (args.metric == 'pck'):
        # create a threshold range
        threshold_range = np.linspace(0.005, 0.1, num=200)
        res = np.zeros((number_of_scenes, len(threshold_range)))

    # loop over scenes (1-2, 1-3, 1-4, 1-5, 1-6)
    for id, k in enumerate(range(2, number_of_scenes + 2)):
        test_dataset = \
            HPatchesDataset(csv_file=osp.join(args.csv_path,
                                              'hpatches_1_{}.csv'.format(k)),
                            image_path_orig=args.image_data_path,
                            transforms=dataset_transforms)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=4)

        if (args.metric == 'aepe'):
            epe_arr = calculate_epe_hpatches(net,
                                             test_dataloader,
                                             device)
            res.append(np.mean(epe_arr))

        if (args.metric == 'pck'):
            for t_id, threshold in enumerate(threshold_range):
                res[id, t_id] = calculate_pck_hpatches(net,
                                                       test_dataloader,
                                                       device,
                                                       alpha=threshold)

    print(res)
