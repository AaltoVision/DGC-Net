
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim


import argparse
import time
import fnmatch
import os.path as osp

from model.net import DGCNet
from utils import calculate_epe_hpatches


def find_checkpoint(path, pattern='*.pth'):
    for _, _, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return name
    return None


# Argument parsing
parser = argparse.ArgumentParser(description='DGC-Net')
# Paths
parser.add_argument('--csv-path', type=str, default='hpathces/data',
                    help='path to training transformation csv folder')
parser.add_argument('--image-path', type=str, default='/hdd/projects/widebaseline_correspondence_search/from_ssd/wide_baseline_correspondence/datasets/dtu/rect_240x240',
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

net.load_state_dict(torch.load(checkpoint_fname)['state_dict'])
net = nn.DataParallel(net)
net.eval()
net.cuda()

with torch.no_grad():
    number_of_scenes = 5
    if (metric == 'AEPE'):
        res = []
        jac = []
    if (metric == 'PCK'):
        threshold_range = np.linspace(0.005, 0.1, num=200)
        res = np.zeros((number_of_scenes, len(threshold_range)))

    # loop over scenes (1-2, 1-3, 1-4, 1-5, 1-6)
    for id, k in enumerate(range(2, number_of_scenes+2)):
        test_dataset = HPatchTestDataset(csv_file=osp.join(args.csv_path, 'hpatches_1_' + str(k) + '.csv'),
                                         image_path=args.image_path,
                                         transforms=dataset_transforms)
        '''
        test_dataset = HPatchTestDataset__M(csv_file=os.path.join(CSV_PATH, 'hpatches_1_'+str(k)+'.csv'),
                                         image_path_orig='/ssd_storage/projects/wide_baseline_correspondence/datasets/HPatches/hpatches-geometry',
                                         image_path=IMG_PATH,
                                         transforms=dataset_transforms)
        '''
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=4)

        if (metric == 'AEPE'):
            epe_arr = calculate_epe_hpatches(net, test_dataloader)
            res.append(np.mean(epe_arr))

        if (metric == 'PCK'):
            for t_id, threshold in enumerate(threshold_range):
                res[id, t_id] = calculate_pck_hpatches(net, test_dataloader, alpha=threshold)

    print(res)