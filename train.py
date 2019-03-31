import numpy as np
import argparse
import time
import os, fnmatch
from os import path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.optim.lr_scheduler as lr_scheduler

from data.dataset import HomoAffTpsDataset #, SUN3DDataset, HomographyDataset, HomoAffPascalTMDataset
from util.loss import L1LossMasked
from util.train_eval_routine import train_epoch, validate_epoch

from model.net import DGCNet

import gc
from tqdm import tqdm

from scipy.misc import toimage
from termcolor import cprint, colored

import pickle
from PIL import Image



# Argument parsing
parser = argparse.ArgumentParser(description='DGC-Net train script')
# Paths
parser.add_argument('--data-path', type=str, default='./data',
                    help='path to TokyoTimeMachine dataset and csv files')
parser.add_argument('--model', type=str, default='dgc', help='Model to use', choices=['dgc', 'dgcm'])
parser.add_argument('--snapshots', type=str, default='./snapshots')
parser.add_argument('--logs', type=str, default='./logs')
# Optimization parameters
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float,
                    default=0.9, help='momentum constant')
parser.add_argument('--start_epoch', type=int, default=-1, help='start epoch')
parser.add_argument('--n_epoch', type=int, default=70, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=32, help='training batch size')
parser.add_argument('--n_threads', type=int, default=8, help='number of parallel threads for dataloaders')
parser.add_argument('--weight-decay', type=float, default=0.00001, help='weight decay constant')
parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')

args = parser.parse_args()

if not os.path.isdir(args.snapshots):
    os.mkdir(args.snapshots)

cur_snapshot = time.strftime('%Y_%m_%d_%H_%M')

if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
    os.mkdir(osp.join(args.snapshots, cur_snapshot))

with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
    pickle.dump(args, f)

cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run with --nocuda")

device = torch.device("cuda" if cuda else "cpu")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

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

train_dataset = HomoAffTpsDataset(image_path=args.data_path,
                                  csv_file=osp.join(args.data_path, csv, 'homo_aff_tps_train.csv'),
                                  transforms=dataset_transforms,
                                  pyramid_param=pyramid_param)

val_dataset = HomoAffTpsDataset(image_path=args.data_path,
                                csv_file=osp.join(args.data_path, csv, 'homo_aff_tps_test.csv'),
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
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
# Scheduler
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2,15,30,45,60], gamma=0.1)
# Criterions
criterion_grid = L1LossMasked().to(device)
criterion_matchability = nn.BCEWithLogitsLoss().to(device)

train_losses = []
val_losses = []
prev_model = None

train_started = time.time()

for epoch in range(args.n_epoch):
    scheduler.step()
    # Training one epoch
    train_loss = train_epoch(model, optimizer, train_dataloader, criterion_grid=criterion_grid, criterion_matchability=criterion_matchability, loss_grid_weights=weights_loss_coeffs)
    train_losses.append(train_loss)
    print(colored('==> ', 'green') + 'Train average loss:', train_loss)

    # Validation
    running_loss_grid, aepe_arrays_240x240 = validate_epoch(model, val_dataloader, criterion_grid=criterion_grid, criterion_matchability=criterion_matchability, loss_grid_weights=weights_loss_coeffs)
    total_val_loss = running_loss_grid #+ running_loss_feat + running_loss_image
    print(colored('==> ', 'blue') + 'Val average grid loss :', running_loss_grid)
    #print(colored('==> ', 'blue') + 'Val average feat loss :', running_loss_feat)
    #print(colored('==> ', 'blue') + 'Val average image loss :', running_loss_image)
    #print(colored('==> ', 'blue') + 'Val average grid masked loss :', running_loss_masked_grid)
    for i in range(len(aepe_arrays_240x240)):
        print(colored('==> ', 'blue') + 'layer {}: mean AEPE_240x240: {}'.format(i, np.mean(aepe_arrays_240x240[i])))
    print(colored('==> ', 'blue') + 'epoch :', epoch+1)

    val_losses.append(total_val_loss)

    writer.add_scalars('Losses', {'train':train_loss, 'val':total_val_loss}, epoch)
    #writer.add_scalars('AEPEs', {'l0':aepe_by_layer[0], 'l1':aepe_by_layer[1], 'l2':aepe_by_layer[2], 'l3':aepe_by_layer[3], 'l4':aepe_by_layer[4]}, epoch)
    #writer.add_scalars('AEPEs_orig',    {'l0':aepe_by_layer_orig[0],    'l1':aepe_by_layer_orig[1],    'l2':aepe_by_layer_orig[2],    'l3':aepe_by_layer_orig[3]}, epoch)
    writer.add_scalars('AEPEs_240x240', {'l0':np.mean(aepe_arrays_240x240[0]), 'l1':np.mean(aepe_arrays_240x240[1]), 'l2':np.mean(aepe_arrays_240x240[2]), 'l3':np.mean(aepe_arrays_240x240[3]), 'l4':np.mean(aepe_arrays_240x240[-1])}, epoch)
    
    # Not clear if we need it after integrating tensorboard
    np.save(osp.join(args.snapshots, cur_snapshot, 'logs.npy'), 
            [train_losses,val_losses])

    if epoch > args.start_epoch:
        # We will be saving only the snapshot which has lowest loss value on the validation set
        cur_snapshot_name = osp.join(args.snapshots, cur_snapshot, 'epoch_{}.pth'.format(epoch + 1))
        if prev_model is None:
            torch.save({'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()}, cur_snapshot_name)
            prev_model = cur_snapshot_name
            best_val = running_loss_grid
        else:
            if running_loss_grid < best_val:
                os.remove(prev_model)
                best_val = running_loss_grid
                print('Saved snapshot:',cur_snapshot_name)
                torch.save({'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()}, cur_snapshot_name)
                prev_model = cur_snapshot_name

gc.collect()
print(args.seed, 'Training took:', time.time()-train_started, 'seconds')
