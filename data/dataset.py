import torch
import cv2

import os
import os.path as osp



class HPatchesDataset(Dataset):
    def __init__(self, csv_file, image_path, transforms, image_size=(240, 240)):
        self.df = pd.read_csv(csv_file)
        self.image_path = image_path
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        obj = str(data.obj)
        im1_id, im2_id = str(data.im1), str(data.im2)

        h_scale, w_scale = self.image_size[0], self.image_size[1]
        h_orig, w_orig = data.Him.astype('int'), data.Wim.astype('int')

        # Homography matrix
        H = data[5:].astype('double').values.reshape((3, 3))

        '''
        As GT homography is calculated for (h_orig, w_orig) images,
        we need to map it to (h_scale, w_scale)
        '''
        S = np.array([[w_scale / w_orig, 0, 0], [0, h_scale / h_orig, 0], [0, 0, 1]])

        H_scale = np.dot(np.dot(S, H), np.linalg.inv(S))

        # inverse homography matrix
        H_inv = np.linalg.inv(H_scale)

        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale), np.linspace(0, h_scale - 1, h_scale))
        X, Y = X.flatten(), Y.flatten()

        # create matrix representation
        XY_hom = np.stack([X, Y, np.ones_like(X)], axis=1).T

        # m
        XY_warp_hom = np.dot(H_inv, XY_hom)

        # vector representation
        X_warp_hom = torch.from_numpy(XY_warp_hom[0, :].float()
        Y_warp_hom = torch.from_numpy(XY_warp_hom[1, :].float()
        Z_warp_hom = torch.from_numpy(XY_warp_hom[2, :].float()

        X_warp = (2 * X_warp_hom / (Z_warp_hom + 1e-8) / (w_scale - 1) - 1).view(h_scale, w_scale)
        Y_warp = (2 * Y_warp_hom / (Z_warp_hom + 1e-8) / (h_scale - 1) - 1).view(h_scale, w_scale)
        
        # GT grid
        grid_gt = torch.stack([X_warp, Y_warp], dim=-1)

        # mask
        mask = grid_gt.ge(-1) & grid_gt.le(1)
        mask = mask[:, :, 0] & mask[:, :, 1]

        img_1 = cv2.imread(osp.join(self.image_path, obj, im1_id + '.ppm'), -1)
        img_2 = cv2.imread(osp.join(self.image_path, obj, im2_id + '.ppm'), -1)

        _, _, ch = img1.shape
        if ch == 3:
            img_1 = cv2.cvtColor(cv2.imread(osp.join(self.image_path, obj, im1_id + '.ppm'), -1), cv2.COLOR_BGR2RGB)
            img_2 = cv2.cvtColor(cv2.imread(osp.join(self.image_path, obj, im2_id + '.ppm'), -1), cv2.COLOR_BGR2RGB)

        # global transforms
        img_1 = self.transforms(img_1)
        img_2 = self.transforms(img_2)

        return {'source_img': img_1, 'target_img': img_2, 'correspondence_map': grid_gt, 'mask': mask.long()}
