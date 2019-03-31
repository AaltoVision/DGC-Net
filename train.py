from __future__ import print_function, division
import sys
import argparse
import time
import os, fnmatch
from os.path import exists, join, basename
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

from data.dataset import WideBaselineDataset, SUN3DDataset, HomographyDataset, HomoAffPascalTMDataset
from util.loss import L1LossMasked
from util.train_eval_routine import train_epoch, validate_epoch
#from model.model import WBCNet
from model.model_v2 import WBCNet

import gc
from tqdm import tqdm # progress bar

from scipy.misc import toimage
from termcolor import cprint, colored

import pickle
from PIL import Image
import numpy as np
from tensorboardX import SummaryWriter


def find_checkpoint(path, pattern='*.pth'):
    for _, _, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return name
    return None


# Argument parsing
parser = argparse.ArgumentParser(description='DGC-Net train script')
# Paths
parser.add_argument('--src_path', type=str, default='/ssd_storage/projects/wide_baseline_correspondence/datasets/demon_data/demon/examples',
                    help='path to training transformation csv folder')
#parser.add_argument('--snapshots', type=str, default='./snapshots/fine_tune_ign_demon_data')
parser.add_argument('--snapshots', type=str, default='./snapshots/aff_TM_no_mm_EQUAL_WEIGHT_COEFFS')
#parser.add_argument('--resume_path', type=str, default='./snapshots/T_2018_04_15_12_03', help='fine tuning')
parser.add_argument('--resume_path', type=str, default='./snapshots/aff_tps_data_match', help='fine tuning')
parser.add_argument('--logs',      type=str, default='./logs/')
# Optimization parameters
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float,
                    default=0.9, help='momentum constant')
parser.add_argument('--start_epoch', type=int, default=-1, help='start epoch')
parser.add_argument('--n_epoch', type=int, default=70,
                    help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size')
parser.add_argument('--n_threads', type=int, default=8,
                    help='number of parallel threads for dataloaders')
parser.add_argument('--weight-decay', type=float,
                    default=0.00001, help='weight decay constant')
parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if not os.path.isdir(args.snapshots):
    os.mkdir(args.snapshots)

cur_snapshot = time.strftime('%Y_%m_%d_%H_%M')

if not os.path.isdir(os.path.join(args.snapshots, cur_snapshot)):
    os.mkdir(os.path.join(args.snapshots, cur_snapshot))

with open(os.path.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
    pickle.dump(args, f)

use_cuda = torch.cuda.is_available()

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


#IMG_PATH = '/ssd_storage/projects/wide_baseline_correspondence/datasets/pascal-voc11'
IMG_PATH = '/ssd_storage/projects/wide_baseline_correspondence/datasets/tokyo_tm'
#IMG_PATH = '/tmp/melekhov'
'''
HEAD_PATH = '/ssd_storage/projects/wide_baseline_correspondence/datasets'

train_dataset = HomoAffPascalTMDataset(head_path=HEAD_PATH, \
                                    csv_file=os.path.join(HEAD_PATH, 'homo_aff_pascal_tm_train_1984.csv'), \
                                    transforms=dataset_transforms)
val_dataset = HomoAffPascalTMDataset(head_path=HEAD_PATH, \
                                    csv_file=os.path.join(HEAD_PATH, 'homo_aff_pascal_tm_test_1984.csv'), \
                                    transforms=dataset_transforms)
'''

train_dataset = WideBaselineDataset(image_path=IMG_PATH, \
                                    #csv_data_file=os.path.join(IMG_PATH, 'ignacio_pascal_aff_train_new_ordering.csv'), \
                                    csv_data_file=os.path.join(IMG_PATH, 'ignacio_tokyo_aff_train_new_ordering.csv'), \
                                    transforms=dataset_transforms, \
                                    joint_aff_tps=False, pyramid_param=pyramid_param)

val_dataset = WideBaselineDataset(image_path=IMG_PATH, \
                                    #csv_data_file=os.path.join(IMG_PATH, 'ignacio_pascal_aff_test_new_ordering.csv'), \
                                    csv_data_file=os.path.join(IMG_PATH, 'ignacio_tokyo_aff_test_new_ordering.csv'), \
                                    transforms=dataset_transforms, \
                                    joint_aff_tps=False, pyramid_param=pyramid_param)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.n_threads)

val_dataloader = DataLoader(val_dataset, batch_size=1,
                              shuffle=False, num_workers=args.n_threads)

'''
train_dataset = HomographyDataset(image_path=IMG_PATH, \
                                    csv_file=os.path.join(IMG_PATH, 'homography_tm_train_768x576.csv'), \
                                    transforms=dataset_transforms, \
                                    pyramid_param=pyramid_param)

val_dataset = HomographyDataset(image_path=IMG_PATH, \
                                    csv_file=os.path.join(IMG_PATH, 'homography_tm_test_768x576.csv'), \
                                    transforms=dataset_transforms, \
                                    pyramid_param=pyramid_param)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.n_threads)

val_dataloader = DataLoader(val_dataset, batch_size=1,
                              shuffle=False, num_workers=args.n_threads)

'''


'''
dataset_vis_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

train_dataset_pure = WideBaselineDataset(image_path=IMG_PATH, \
                                    csv_data_file=os.path.join(IMG_PATH, 'rocco_aff_tps_train_1984.csv'), \
                                    transforms=dataset_vis_transforms, \
                                    joint_aff_tps=True, pyramid_param=pyramid_param)

imgs_container = []
imgs_container.append(train_dataset[IMG_ID]['source_image'].squeeze().transpose(0,1).transpose(1,2))
#print(train_dataset[IMG_ID]['source_image'].squeeze().transpose(0,1).transpose(1,2))
#sys.exit()
imgs_container.append(train_dataset[IMG_ID]['target_image'].squeeze().transpose(0,1).transpose(1,2))
aff = train_dataset[IMG_ID]['theta_gt']
maps_pyro = train_dataset[IMG_ID]['correspondence_map_pyro']
print(maps_pyro[-1].shape)

fig = plt.figure(figsize=(12, 3))

fig_titles = ['source_image', 'target_image']
rows, cols = (1, 2)

for k, (title, img) in enumerate(zip(fig_titles, imgs_container)):
    ax = fig.add_subplot(rows, cols, k+1)
    ax.set_title(title)
    ax.imshow(img)

plt.tight_layout()
plt.show()
'''



# self, src_path, imgs_left_npy, imgs_right_npy, grid_npy, motion_npy, pyramid_param, transforms
'''
# Dataset and dataloader
train_dataset = SUN3DDataset(src_path=os.path.join(args.src_path, 'train_data'),
                             imgs_left_npy='ign_sun3d_rgbd_mvs_train_all_imgs_left_RND.npy',
                             imgs_right_npy='ign_sun3d_rgbd_mvs_train_all_imgs_right_RND.npy',
                             grid_npy='ign_sun3d_rgbd_mvs_train_all_grid_RND.npy',
                             motion_npy='sun3d_all_motion_RND.npy',
                             pyramid_param=pyramid_param, transforms=dataset_transforms)

val_dataset   = SUN3DDataset(src_path=os.path.join(args.src_path, 'test_data'),
                             imgs_left_npy='ign_sun3d_rgbd_mvs_test_all_imgs_left_RND.npy',
                             imgs_right_npy='ign_sun3d_rgbd_mvs_test_all_imgs_right_RND.npy',
                             grid_npy='ign_sun3d_rgbd_mvs_test_all_grid_RND.npy',
                             motion_npy='sun3d_all_test_motion_RND.npy',
                             pyramid_param=pyramid_param, transforms=dataset_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.n_threads)

val_dataloader = DataLoader(val_dataset, batch_size=1,
                              shuffle=False, num_workers=args.n_threads)
'''

#checkpoint_name = find_checkpoint(args.resume_path)
#checkpoint = torch.load(os.path.join(args.resume_path, checkpoint_name))
model = WBCNet()
#print(model)
#sys.exit()
#model.load_state_dict(checkpoint['state_dict'])
model = nn.DataParallel(model)
model.cuda()


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[3,15,30,45,60], gamma=0.1)
#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 40], gamma=0.1)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2,15,30,45,60], gamma=0.1)

#criterion_grid = nn.MSELoss().cuda()
#criterion_grid = nn.L1Loss().cuda()
criterion_grid = L1LossMasked().cuda()
criterion_matchability = nn.BCEWithLogitsLoss().cuda()

train_losses = []
val_losses = []
prev_model = None

train_started = time.time()

writer = SummaryWriter(os.path.join(args.logs,'wide-baseline-correspondence',cur_snapshot))

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
    #mean_aepes.append(mean_aepe)
    #sys.exit()
    
    writer.add_scalars('Losses', {'train':train_loss, 'val':total_val_loss}, epoch)
    #writer.add_scalars('AEPEs', {'l0':aepe_by_layer[0], 'l1':aepe_by_layer[1], 'l2':aepe_by_layer[2], 'l3':aepe_by_layer[3], 'l4':aepe_by_layer[4]}, epoch)
    #writer.add_scalars('AEPEs_orig',    {'l0':aepe_by_layer_orig[0],    'l1':aepe_by_layer_orig[1],    'l2':aepe_by_layer_orig[2],    'l3':aepe_by_layer_orig[3]}, epoch)
    writer.add_scalars('AEPEs_240x240', {'l0':np.mean(aepe_arrays_240x240[0]), 'l1':np.mean(aepe_arrays_240x240[1]), 'l2':np.mean(aepe_arrays_240x240[2]), 'l3':np.mean(aepe_arrays_240x240[3]), 'l4':np.mean(aepe_arrays_240x240[-1])}, epoch)
    
    # Not clear if we need it after integrating tensorboard
    np.save(os.path.join(args.snapshots, cur_snapshot, 'logs.npy'), 
            [train_losses,val_losses])

    if epoch > args.start_epoch:
        # We will be saving only the snapshot which has lowest loss value on the validation set
        cur_snapshot_name = os.path.join(args.snapshots, cur_snapshot, 'epoch_{}.pth'.format(epoch+1))
        if prev_model is None:
            torch.save({'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()}, cur_snapshot_name)
            prev_model = cur_snapshot_name
            #best_val = total_val_loss
            best_val = running_loss_grid
        else:
            #if total_val_loss < best_val:
            if running_loss_grid < best_val:
                os.remove(prev_model)
                #best_val= total_val_loss
                best_val = running_loss_grid
                print('Saved snapshot:',cur_snapshot_name)
                torch.save({'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()}, cur_snapshot_name)
                prev_model = cur_snapshot_name

gc.collect()
print(args.seed, 'Training took:', time.time()-train_started, 'seconds')
