import numpy as np
import cv2

from tqdm import tqdm
import torch
import torch.nn.functional as F


def train_epoch(net,
                optimizer,
                train_loader,
                device,
                criterion_grid,
                criterion_matchability=None,
                loss_grid_weights=None,
                L_coeff=1):
    """
    Training epoch script
    Args:
        net: model architecture
        optimizer: optimizer to be used for traninig `net`
        train_loader: dataloader
        device: `cpu` or `gpu`
        criterion_grid: criterion for esimation pixel correspondence (L1Masked)
        criterion_matchability: criterion for mask optimization
        loss_grid_weights: weight coefficients for each grid estimates tensor
            for each level of the feature pyramid
        L_coeff: weight coefficient to balance `criterion_grid` and
            `criterion_matchability`
    Output:
        running_total_loss: total training loss
    """

    net.train()
    running_total_loss = 0
    running_match_loss = 0
    if loss_grid_weights is None:
        loss_grid_weights = [1, 1, 1, 1, 1]

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:

        optimizer.zero_grad()

        # net predictions
        estimates_grid, estimates_mask = \
            net(mini_batch['source_image'].to(device),
                mini_batch['target_image'].to(device))

        if criterion_matchability is not None and estimates_mask is None:
            raise ValueError('Cannot use `criterion_matchability` \
                without mask estimates')

        Loss_masked_grid = 0
        EPE_loss = 0

        # grid loss components (over all layers of the feature pyramid):
        for k in range(0, len(estimates_grid)):

            grid_gt = mini_batch['correspondence_map_pyro'][k].to(device)
            bs, s_x, s_y, _ = grid_gt.shape

            flow_est = estimates_grid[k].permute(0, 2, 3, 1)
            flow_target = grid_gt

            # calculating mask
            mask_x_gt = \
                flow_target[:, :, :, 0].ge(-1) & flow_target[:, :, :, 0].le(1)
            mask_y_gt = \
                flow_target[:, :, :, 1].ge(-1) & flow_target[:, :, :, 1].le(1)
            mask_gt = mask_x_gt & mask_y_gt

            # number of valid pixels based on the mask
            N_valid_pxs = mask_gt.view(1, bs * s_x * s_y).data.sum()

            # applying mask
            mask_gt = torch.cat((mask_gt.unsqueeze(3),
                                 mask_gt.unsqueeze(3)), dim=3).float()
            flow_target_m = flow_target * mask_gt
            flow_est_m = flow_est * mask_gt

            # compute grid loss
            Loss_masked_grid = Loss_masked_grid + \
                loss_grid_weights[k] * criterion_grid(flow_est_m,
                                                      flow_target_m,
                                                      N_valid_pxs)

        Loss_match = 0
        if estimates_mask is not None:
            match_mask_gt = \
                mini_batch['mask_x'][-1].to(device) & \
                mini_batch['mask_y'][-1].to(device)
            Loss_match = \
                criterion_matchability(estimates_mask.squeeze(1),
                                       match_mask_gt)

        Loss = Loss_masked_grid + L_coeff * Loss_match
        Loss.backward()

        optimizer.step()

        running_total_loss += Loss.item()
        if estimates_mask is not None:
            running_match_loss += Loss_match.item()
            pbar.set_description('R_total_loss: %.3f/%.3f | \
                Match_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                          Loss.item(),
                                          running_match_loss / (i + 1),
                                          Loss_match.item()))
        else:
            pbar.set_description(
                'R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                             Loss.item()))

    running_total_loss /= len(train_loader)
    return running_total_loss


def validate_epoch(net,
                   val_loader,
                   device,
                   criterion_grid,
                   criterion_matchability=None,
                   loss_grid_weights=None,
                   L_coeff=1):
    """
    Validation epoch script
    Args:
        net: model architecture
        val_loader: dataloader
        device: `cpu` or `gpu`
        criterion_grid: criterion for esimation pixel correspondence (L1Masked)
        criterion_matchability: criterion for mask optimization
        loss_grid_weights: weight coefficients for each grid estimates tensor
            for each level of the feature pyramid
        L_coeff: weight coefficient to balance `criterion_grid` and
            `criterion_matchability`
    Output:
        running_total_loss: total validation loss
    """

    net.eval()
    if loss_grid_weights:
        loss_grid_weights = [1, 1, 1, 1, 1]

    bilinear_coeffs = [16, 8, 4, 2, 1]
    aepe_arrays_240x240 = [[] for _ in bilinear_coeffs]
    running_total_loss = 0
    running_match_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, mini_batch in pbar:
            # net predictions
            estimates_grid, estimates_mask = \
                net(mini_batch['source_image'].to(device),
                    mini_batch['target_image'].to(device))

            if criterion_matchability is not None and estimates_mask is None:
                raise ValueError('Cannot use criterion_matchability \
                    without mask estimates')

            Loss_masked_grid = 0
            # grid loss components (over all layers of the feature pyramid):
            for k in range(0, len(estimates_grid)):
                grid_gt = mini_batch['correspondence_map_pyro'][k].to(device)
                bs, s_x, s_y, _ = grid_gt.shape

                flow_est = estimates_grid[k].permute(0, 2, 3, 1)
                flow_target = grid_gt

                # calculating mask
                mask_x_gt = flow_target[:, :, :, 0].ge(-1) & \
                    flow_target[:, :, :, 0].le(1)
                mask_y_gt = flow_target[:, :, :, 1].ge(-1) & \
                    flow_target[:, :, :, 1].le(1)
                mask_gt = mask_x_gt & mask_y_gt

                # number of valid pixels based on the mask
                N_valid_pxs = mask_gt.view(1, bs * s_x * s_y).data.sum()

                # applying mask
                mask_gt = torch.cat((mask_gt.unsqueeze(3),
                                     mask_gt.unsqueeze(3)), dim=3).float()
                flow_target_m = flow_target * mask_gt
                flow_est_m = flow_est * mask_gt

                # compute grid loss
                Loss_masked_grid = Loss_masked_grid + \
                    loss_grid_weights[k] * criterion_grid(flow_est_m,
                                                          flow_target_m,
                                                          N_valid_pxs)

            # matchability mask loss
            Loss_match = 0
            if estimates_mask is not None:
                match_mask_gt = mini_batch['mask_x'][-1].to(device) & \
                    mini_batch['mask_y'][-1].to(device)
                Loss_match = criterion_matchability(estimates_mask.squeeze(1),
                                                    match_mask_gt)

            Loss = Loss_masked_grid + L_coeff * Loss_match

            running_total_loss += Loss.item()
            if estimates_mask is not None:
                running_match_loss += Loss_match.item()
                pbar.set_description('R_total_loss: %.3f/%.3f | \
                    Match_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                              Loss.item(),
                                              running_match_loss / (i + 1),
                                              Loss_match.item()))
            else:
                pbar.set_description(
                    'R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                                 Loss.item()))

    return running_total_loss / len(val_loader)
