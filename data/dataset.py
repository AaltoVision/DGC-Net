from os import path as osp

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def center_crop(img, size):
    """
    Get the center crop of the input image
    Args:
        img: input image [BxCxHxW]
        size: size of the center crop (tuple)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    """

    if not isinstance(size, tuple):
        size = (size, size)

    img = img.copy()
    w, h = img.shape[1::-1]

    pad_w = 0
    pad_h = 0
    if w < size[0]:
        pad_w = np.uint16((size[0] - w) / 2)
    if h < size[1]:
        pad_h = np.uint16((size[1] - h) / 2)
    img_pad = cv2.copyMakeBorder(img,
                                 pad_h,
                                 pad_h,
                                 pad_w,
                                 pad_w,
                                 cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    w, h = img_pad.shape[1::-1]

    x1 = w // 2 - size[0] // 2
    y1 = h // 2 - size[1] // 2

    img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0], :]

    return img_pad, x1, y1


class HPatchesDataset(Dataset):
    """
    HPatches dataset (for evaluation)
    Args:
        csv_file: csv file with ground-truth data
        image_path_orig: filepath to the dataset (full resolution)
        transforms: image transformations (data preprocessing)
        image_size: size (tuple) of the output images
    Output:
        source_image: source image
        target_image: target image
        correspondence_map: pixel correspondence map
            between source and target views
        mask: valid/invalid correspondences
    """

    def __init__(self,
                 csv_file,
                 image_path_orig,
                 transforms,
                 image_size=(240, 240)):
        self.df = pd.read_csv(csv_file)
        self.image_path_orig = image_path_orig
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        obj = str(data.obj)
        im1_id, im2_id = str(data.im1), str(data.im2)
        h_scale, w_scale = self.image_size[0], self.image_size[1]

        h_ref_orig, w_ref_orig = data.Him.astype('int'), data.Wim.astype('int')
        h_trg_orig, w_trg_orig, _ = \
            cv2.imread(osp.join(self.image_path_orig,
                                obj,
                                im2_id + '.ppm'), -1).shape

        H = data[5:].astype('double').values.reshape((3, 3))

        '''
        As gt homography is calculated for (h_orig, w_orig) images,
        we need to
        map it to (h_scale, w_scale)
        H_scale = S * H * inv(S)
        '''
        S1 = np.array([[w_scale / w_ref_orig, 0, 0],
                       [0, h_scale / h_ref_orig, 0],
                       [0, 0, 1]])
        S2 = np.array([[w_scale / w_trg_orig, 0, 0],
                       [0, h_scale / h_trg_orig, 0],
                       [0, 0, 1]])

        H_scale = np.dot(np.dot(S2, H), np.linalg.inv(S1))

        # inverse homography matrix
        Hinv = np.linalg.inv(H_scale)

        # estimate the grid
        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                           np.linspace(0, h_scale - 1, h_scale))
        X, Y = X.flatten(), Y.flatten()

        # create matrix representation
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

        # multiply Hinv to XYhom to find the warped grid
        XYwarpHom = np.dot(Hinv, XYhom)

        # vector representation
        XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
        YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
        ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

        Xwarp = \
            (2 * XwarpHom / (ZwarpHom + 1e-8) / (w_scale - 1) - 1)
        Ywarp = \
            (2 * YwarpHom / (ZwarpHom + 1e-8) / (h_scale - 1) - 1)
        # and now the grid
        grid_gt = torch.stack([Xwarp.view(h_scale, w_scale),
                               Ywarp.view(h_scale, w_scale)], dim=-1)

        # mask
        mask = grid_gt.ge(-1) & grid_gt.le(1)
        mask = mask[:, :, 0] & mask[:, :, 1]

        img1 = \
            cv2.resize(cv2.imread(osp.join(self.image_path_orig,
                                           obj,
                                           im1_id + '.ppm'), -1),
                       self.image_size)
        img2 = \
            cv2.resize(cv2.imread(osp.join(self.image_path_orig,
                                           obj,
                                           im2_id + '.ppm'), -1),
                       self.image_size)
        _, _, ch = img1.shape
        if ch == 3:
            img1_tmp = cv2.imread(osp.join(self.image_path_orig,
                                           obj,
                                           im1_id + '.ppm'), -1)
            img2_tmp = cv2.imread(osp.join(self.image_path_orig,
                                           obj,
                                           im2_id + '.ppm'), -1)
            img1 = cv2.cvtColor(cv2.resize(img1_tmp,
                                           self.image_size),
                                cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.resize(img2_tmp,
                                           self.image_size),
                                cv2.COLOR_BGR2RGB)

        # global transforms
        img1 = self.transforms(img1)
        img2 = self.transforms(img2)

        return {'source_image': img1,
                'target_image': img2,
                'correspondence_map': grid_gt,
                'mask': mask.long()}


class HomoAffTpsDataset(Dataset):
    """
    Main dataset for training/validation the proposed approach.
    It can handle affine, TPS, and Homography transformations.
    Args:
        image_path: filepath to the dataset
            (either TokyoTimeMachine or Pascal-VOC)
        csv_file: csv file with ground-truth data
        transforms: image transformations (data preprocessing)
        pyramid_param: spatial resolution of the feature maps at each level
            of the feature pyramid (list)
        output_size: size (tuple) of the output images
    Output:
        source_image: source image
        target_image: target image
        correspondence_map_pyro: pixel correspondence map
            for each feature pyramid level
        mask_x: X component of the mask (valid/invalid correspondences)
        mask_y: Y component of the mask (valid/invalid correspondences)
    """

    def __init__(self,
                 image_path,
                 csv_file,
                 transforms,
                 pyramid_param=[15, 30, 60, 120, 240],
                 output_size=(240, 240)):
        super().__init__()
        self.img_path = image_path
        self.transform_dict = {0: 'aff', 1: 'tps', 2: 'homo'}
        self.transforms = transforms
        self.pyramid_param = pyramid_param
        self.df = pd.read_csv(csv_file)

        self.H_AFF_TPS, self.W_AFF_TPS = (480, 640)
        self.H_HOMO, self.W_HOMO = (576, 768)
        self.H_OUT, self.W_OUT = (output_size)
        self.THETA_IDENTITY = \
            torch.Tensor(np.expand_dims(np.array([[1, 0, 0],
                                                  [0, 1, 0]]),
                                        0).astype(np.float32))
        self.gridGen = TpsGridGen(self.H_OUT, self.W_OUT)

    def transform_image(self,
                        image,
                        out_h,
                        out_w,
                        padding_factor=1.0,
                        crop_factor=1.0,
                        theta=None):
        sampling_grid = self.generate_grid(out_h, out_w, theta)
        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data * padding_factor * crop_factor
        # sample transformed image
        warped_image_batch = F.grid_sample(image, sampling_grid)
        return warped_image_batch

    def generate_grid(self, out_h, out_w, theta=None):
        out_size = torch.Size((1, 3, out_h, out_w))
        if theta is None:
            theta = self.THETA_IDENTITY
            theta = theta.expand(1, 2, 3).contiguous()
            return F.affine_grid(theta, out_size)
        elif (theta.shape[1] == 2):
            return F.affine_grid(theta, out_size)
        else:
            return self.gridGen(theta)

    def get_grid(self, H, ccrop):
        # top-left corner of the central crop
        X_CCROP, Y_CCROP = ccrop[0], ccrop[1]

        W_FULL, H_FULL = (self.W_HOMO, self.H_HOMO)
        W_SCALE, H_SCALE = (self.W_OUT, self.H_OUT)

        # inverse homography matrix
        Hinv = np.linalg.inv(H)

        # estimate the grid for the whole image
        X, Y = np.meshgrid(np.linspace(0, W_FULL - 1, W_FULL),
                           np.linspace(0, H_FULL - 1, H_FULL))
        X_, Y_ = X, Y
        X, Y = X.flatten(), Y.flatten()

        # create matrix representation
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

        # multiply Hinv to XYhom to find the warped grid
        XYwarpHom = np.dot(Hinv, XYhom)

        # vector representation
        XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
        YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
        ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

        X_grid_pivot = (XwarpHom / (ZwarpHom + 1e-8)).view(H_FULL, W_FULL)
        Y_grid_pivot = (YwarpHom / (ZwarpHom + 1e-8)).view(H_FULL, W_FULL)

        # normalize XwarpHom and YwarpHom and cast to [-1, 1] range
        Xwarp = (2 * X_grid_pivot / (W_FULL - 1) - 1)
        Ywarp = (2 * Y_grid_pivot / (H_FULL - 1) - 1)
        grid_full = torch.stack([Xwarp, Ywarp], dim=-1)

        # getting the central patch from the pivot
        Xwarp_crop = X_grid_pivot[Y_CCROP:Y_CCROP + H_SCALE,
                                  X_CCROP:X_CCROP + W_SCALE]
        Ywarp_crop = Y_grid_pivot[Y_CCROP:Y_CCROP + H_SCALE,
                                  X_CCROP:X_CCROP + W_SCALE]
        X_crop = X_[Y_CCROP:Y_CCROP + H_SCALE,
                    X_CCROP:X_CCROP + W_SCALE]
        Y_crop = Y_[Y_CCROP:Y_CCROP + H_SCALE,
                    X_CCROP:X_CCROP + W_SCALE]

        # crop grid
        Xwarp_crop_range = \
            2 * (Xwarp_crop - X_crop.min()) / (X_crop.max() - X_crop.min()) - 1
        Ywarp_crop_range = \
            2 * (Ywarp_crop - Y_crop.min()) / (Y_crop.max() - Y_crop.min()) - 1
        grid_crop = torch.stack([Xwarp_crop_range,
                                 Ywarp_crop_range], dim=-1)
        return grid_full.unsqueeze(0), grid_crop.unsqueeze(0)

    @staticmethod
    def symmetric_image_pad(image_batch, padding_factor):
        """
        Pad an input image mini-batch symmetrically
        Args:
            image_batch: an input image mini-batch to be pre-processed
            padding_factor: padding factor
        Output:
            image_batch: padded image mini-batch
        """
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w - 1, -1, -1))
        idx_pad_right = torch.LongTensor(range(w - 1, w - pad_w - 1, -1))
        idx_pad_top = torch.LongTensor(range(pad_h - 1, -1, -1))
        idx_pad_bottom = torch.LongTensor(range(h - 1, h - pad_h - 1, -1))

        image_batch = torch.cat((image_batch.index_select(3, idx_pad_left),
                                 image_batch,
                                 image_batch.index_select(3, idx_pad_right)),
                                3)
        image_batch = torch.cat((image_batch.index_select(2, idx_pad_top),
                                 image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)),
                                2)
        return image_batch

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        # get the transformation type flag
        transform_type = data['aff/tps/homo'].astype('uint8')

        # aff/tps transformations
        if transform_type == 0 or transform_type == 1:
            # read image
            source_img_name = osp.join(self.img_path, data.fname)
            source_img = cv2.cvtColor(cv2.imread(source_img_name),
                                      cv2.COLOR_BGR2RGB)

            if transform_type == 0:
                theta = data.iloc[2:8].values.astype('float').reshape(2, 3)
                theta = torch.Tensor(theta.astype(np.float32)).expand(1, 2, 3)
            else:
                theta = data.iloc[2:].values.astype('float')
                theta = np.expand_dims(np.expand_dims(theta, 1), 2)
                theta = torch.Tensor(theta.astype(np.float32))
                theta = theta.expand(1, 18, 1, 1)

            # make arrays float tensor for subsequent processing
            image = torch.Tensor(source_img.astype(np.float32))

            if image.numpy().ndim == 2:
                image = \
                    torch.Tensor(np.dstack((source_img.astype(np.float32),
                                            source_img.astype(np.float32),
                                            source_img.astype(np.float32))))

            image = image.transpose(1, 2).transpose(0, 1)

            # Resize image using bilinear sampling with identity affine
            # image is 480x640
            image = self.transform_image(image.unsqueeze(0),
                                         self.H_AFF_TPS,
                                         self.W_AFF_TPS)

            # generate symmetrically padded image for bigger sampling region
            image_pad = self.symmetric_image_pad(image, padding_factor=0.5)

            # get cropped source image (240x240)
            img_src_crop = \
                self.transform_image(image_pad,
                                     self.H_OUT,
                                     self.W_OUT,
                                     padding_factor=0.5,
                                     crop_factor=9 / 16).squeeze().numpy()

            # get cropped target image (240x240)
            img_target_crop = \
                self.transform_image(image_pad,
                                     self.H_OUT,
                                     self.W_OUT,
                                     padding_factor=0.5,
                                     crop_factor=9 / 16,
                                     theta=theta).squeeze().numpy()

            # convert to [H, W, C] convention (for np arrays)
            img_src_crop = img_src_crop.transpose((1, 2, 0))
            img_target_crop = img_target_crop.transpose((1, 2, 0))

        # Homography transformation
        elif transform_type == 2:
            # Homography matrix for 768x576 image resolution
            theta = data.iloc[2:11].values.astype('double').reshape(3, 3)

            img_src_orig = \
                cv2.cvtColor(cv2.resize(cv2.imread(osp.join(self.img_path,
                                                            data.fname)),
                                        None,
                                        fx=1.2,
                                        fy=1.2,
                                        interpolation=cv2.INTER_LINEAR),
                             cv2.COLOR_BGR2RGB)

            # get a central crop:
            img_src_crop, x1_crop, y1_crop = center_crop(img_src_orig,
                                                         self.W_OUT)

            # Obtaining the full and crop grids out of H
            grid_full, grid_crop = self.get_grid(theta,
                                                 ccrop=(x1_crop, y1_crop))

            # warp the fullsize original source image
            img_src_orig = torch.Tensor(img_src_orig.astype(np.float32))
            img_src_orig = img_src_orig.permute(2, 0, 1)
            img_orig_target_vrbl = F.grid_sample(img_src_orig.unsqueeze(0),
                                                 grid_full)
            img_orig_target_vrbl = \
                img_orig_target_vrbl.squeeze().permute(1, 2, 0)

            # get the central crop of the target image
            img_target_crop, _, _ = center_crop(img_orig_target_vrbl.numpy(),
                                                self.W_OUT)

        else:
            print('Error: transformation type')

        if self.transforms is not None:
            cropped_source_image = \
                self.transforms(img_src_crop.astype(np.uint8))
            cropped_target_image = \
                self.transforms(img_target_crop.astype(np.uint8))
        else:
            cropped_source_image = \
                torch.Tensor(img_src_crop.astype(np.float32))
            cropped_target_image = \
                torch.Tensor(img_target_crop.astype(np.float32))

            # convert to [C, H, W] convention (for tensors)
            cropped_source_image = cropped_source_image.permute(-1, 0, 1)
            cropped_target_image = cropped_target_image.permute(-1, 0, 1)

        # consturct a pyramid with a corresponding grid on each layer
        grid_pyramid = []
        mask_x = []
        mask_y = []
        if transform_type == 0:
            for layer_size in self.pyramid_param:
                grid = self.generate_grid(layer_size,
                                          layer_size,
                                          theta).squeeze(0)
                mask = grid.ge(-1) & grid.le(1)
                grid_pyramid.append(grid)
                mask_x.append(mask[:, :, 0])
                mask_y.append(mask[:, :, 1])
        elif transform_type == 1:
            grid = self.generate_grid(self.H_OUT,
                                      self.W_OUT,
                                      theta).squeeze(0)
            for layer_size in self.pyramid_param:
                grid_m = torch.from_numpy(cv2.resize(grid.numpy(),
                                                     (layer_size, layer_size)))
                mask = grid_m.ge(-1) & grid_m.le(1)
                grid_pyramid.append(grid_m)
                mask_x.append(mask[:, :, 0])
                mask_y.append(mask[:, :, 1])
        elif transform_type == 2:
            grid = grid_crop.squeeze(0)
            for layer_size in self.pyramid_param:
                grid_m = torch.from_numpy(cv2.resize(grid.numpy(),
                                                     (layer_size, layer_size)))
                mask = grid_m.ge(-1) & grid_m.le(1)
                grid_pyramid.append(grid_m)
                mask_x.append(mask[:, :, 0])
                mask_y.append(mask[:, :, 1])

        return {'source_image': cropped_source_image,
                'target_image': cropped_target_image,
                'correspondence_map_pyro': grid_pyramid,
                'mask_x': mask_x,
                'mask_y': mask_y}


class TpsGridGen(nn.Module):
    """
    Adopted version of synthetically transformed pairs dataset by I.Rocco
    https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self,
                 out_h=240,
                 out_w=240,
                 use_regular_grid=True,
                 grid_size=3,
                 reg_factor=0,
                 use_cuda=False):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
                                               np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = \
                P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = \
                P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta,
                                                torch.cat((self.grid_X,
                                                           self.grid_Y), 3))
        return warped_grid

    def compute_L_inverse(self, X, Y):
        # num of points (along dim 0)
        N = X.size()[0]

        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = \
            torch.pow(Xmat - Xmat.transpose(0, 1), 2) + \
            torch.pow(Ymat - Ymat.transpose(0, 1), 2)

        # make diagonal 1 to avoid NaN in log computation
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))

        # construct matrix L
        OO = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((OO, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1),
                       torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        '''
        points should be in the [B,H,W,2] format,
        where points[:,:,:,0] are the X coords
        and points[:,:,:,1] are the Y coords
        '''

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        '''
        repeat pre-defined control points along
        spatial dimensions of points to be transformed
        '''
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_X)
        W_Y = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_Y)
        '''
        reshape
        W_X,W,Y: size [B,H,W,1,N]
        '''
        W_X = \
            W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        W_Y = \
            W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        # compute weights for affine part
        A_X = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_X)
        A_Y = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_Y)
        '''
        reshape
        A_X,A,Y: size [B,H,W,1,3]
        '''
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        '''
        compute distance P_i - (grid_X,grid_Y)
        grid is expanded in point dim 4, but not in batch dim 0,
        as points P_X,P_Y are fixed for all batch
        '''
        sz_x = points[:, :, :, 0].size()
        sz_y = points[:, :, :, 1].size()
        p_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4)
        p_X_for_summation = p_X_for_summation.expand(sz_x + (1, self.N))
        p_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4)
        p_Y_for_summation = p_Y_for_summation.expand(sz_y + (1, self.N))

        if points_b == 1:
            delta_X = p_X_for_summation - P_X
            delta_Y = p_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = p_X_for_summation - P_X.expand_as(p_X_for_summation)
            delta_Y = p_Y_for_summation - P_Y.expand_as(p_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        '''
        U: size [1,H,W,1,N]
        avoid NaN in log computation
        '''
        dist_squared[dist_squared == 0] = 1
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) +
                                                   points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) +
                                                   points_Y_batch.size()[1:])

        points_X_prime = \
            A_X[:, :, :, :, 0] + \
            torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = \
            A_Y[:, :, :, :, 0] + \
            torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)
        return torch.cat((points_X_prime, points_Y_prime), 3)
