import numpy as np
import argparse
import time
import random
import os
from os import path as osp
from termcolor import colored
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler

from data.dataset import HomoAffTpsDataset
from utils.loss import L1LossMasked
from utils.optimize import train_epoch, validate_epoch
from model.net import DGCNet


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='DGC-Net train script')
    # Paths
    parser.add_argument('--image-data-path', type=str, default='',
                        help='path to TokyoTimeMachine dataset and csv files')
    parser.add_argument('--metadata-path', type=str, default='./data/',
                        help='path to the CSV files')
    parser.add_argument('--model', type=str, default='dgc',
                        help='Model to use', choices=['dgc', 'dgcm'])
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--logs', type=str, default='./logs')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='momentum constant')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--n_epoch', type=int, default=70,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=8,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='weight decay constant')
    parser.add_argument('--seed', type=int, default=1984,
                        help='Pseudo-RNG seed')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.isdir(args.snapshots):
        os.mkdir(args.snapshots)

    cur_snapshot = time.strftime('%Y_%m_%d_%H_%M')

    if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
        os.mkdir(osp.join(args.snapshots, cur_snapshot))

    with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean_vector = np.array([0.485, 0.456, 0.406])
    std_vector = np.array([0.229, 0.224, 0.225])
    normTransform = transforms.Normalize(mean_vector, std_vector)
    dataset_transforms = transforms.Compose([
            transforms.ToTensor(),
            normTransform
        ])

    pyramid_param = [15, 30, 60, 120, 240]
    weights_loss_coeffs = [1, 1, 1, 1, 1]
    weights_loss_feat = [1, 1, 1, 1]

    train_dataset = \
        HomoAffTpsDataset(image_path=args.image_data_path,
                          csv_file=osp.join(args.metadata_path,
                                            'csv',
                                            'homo_aff_tps_train.csv'),
                          transforms=dataset_transforms,
                          pyramid_param=pyramid_param)

    val_dataset = \
        HomoAffTpsDataset(image_path=args.image_data_path,
                          csv_file=osp.join(args.metadata_path,
                                            'csv',
                                            'homo_aff_tps_test.csv'),
                          transforms=dataset_transforms,
                          pyramid_param=pyramid_param)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.n_threads)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.n_threads)

    # Model
    if args.model == 'dgc':
        model = DGCNet()
        print(colored('==> ', 'blue') + 'DGC-Net created.')
    elif args.model == 'dgcm':
        model = DGCNet(mask=True)
        print(colored('==> ', 'blue') + 'DGC+M-Net created.')
    else:
        raise ValueError('check the model type [dgc, dgcm]')

    model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer
    optimizer = \
        optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=args.lr,
                   weight_decay=args.weight_decay)
    # Scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[2, 15, 30, 45, 60],
                                         gamma=0.1)
    # Criterions
    criterion_grid = L1LossMasked().to(device)
    criterion_match = None
    if args.model == 'dgcm':
        criterion_match = nn.BCEWithLogitsLoss().to(device)

    train_losses = []
    val_losses = []
    prev_model = None

    train_started = time.time()

    for epoch in range(args.n_epoch):
        scheduler.step()
        # Training one epoch
        train_loss = train_epoch(model,
                                 optimizer,
                                 train_dataloader,
                                 device,
                                 criterion_grid=criterion_grid,
                                 criterion_matchability=criterion_match,
                                 loss_grid_weights=weights_loss_coeffs)
        train_losses.append(train_loss)
        print(colored('==> ', 'green') + 'Train average loss:', train_loss)

        # Validation
        val_loss_grid = validate_epoch(model,
                                       val_dataloader,
                                       device,
                                       criterion_grid=criterion_grid,
                                       criterion_matchability=criterion_match,
                                       loss_grid_weights=weights_loss_coeffs)
        print(colored('==> ', 'blue') + 'Val average grid loss :',
              val_loss_grid)
        print(colored('==> ', 'blue') + 'epoch :', epoch + 1)
        val_losses.append(val_loss_grid)

        np.save(osp.join(args.snapshots, cur_snapshot, 'logs.npy'),
                [train_losses, val_losses])

        if epoch > args.start_epoch:
            '''
            We will be saving only the snapshot which
            has lowest loss value on the validation set
            '''
            cur_snapshot_name = osp.join(args.snapshots,
                                         cur_snapshot,
                                         'epoch_{}.pth'.format(epoch + 1))
            if prev_model is None:
                torch.save({'state_dict': model.module.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           cur_snapshot_name)
                prev_model = cur_snapshot_name
                best_val = val_loss_grid
            else:
                if val_loss_grid < best_val:
                    os.remove(prev_model)
                    best_val = val_loss_grid
                    print('Saved snapshot:', cur_snapshot_name)
                    torch.save({'state_dict': model.module.state_dict(),
                                'optimizer': optimizer.state_dict()},
                               cur_snapshot_name)
                    prev_model = cur_snapshot_name

    print(args.seed, 'Training took:', time.time()-train_started, 'seconds')
