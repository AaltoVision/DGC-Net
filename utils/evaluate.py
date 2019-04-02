from tqdm import tqdm
import itertools



def epe(input_flow, target_flow):
    """
    End-point-Error computation
    """
    return torch.norm(target_flow - input_flow, p=2, dim=1).mean()


def correct_correspondences(input_flow, target_flow, alpha, im_size=240):
    input_flow = input_flow.unsqueeze(0)
    target_flow = target_flow.unsqueeze(0)
    dist = torch.norm(target_flow - input_flow,p=2,dim=0).unsqueeze(0)

    pck_threshold = alpha * im_size
    mask = dist.le(pck_threshold)

    return len(dist[mask.detach()])


def calculate_epe_hpatches(net, val_loader, img_size=240):
    aepe_array = []
    n_registered_pxs = 0

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for _, mini_batch in pbar:

        source_img = mini_batch['source_image'].to(net.device())
        target_img = mini_batch['target_image'].to(net.device())
        bs, _, _, _ = source_img.shape

        # net prediction
        estimates_grid, estimates_mask = net(source_img, target_img)

        flow_est = estimates_grid[-1].transpose(1,2).transpose(2,3).to(net.device())
        flow_target = mini_batch['correspondence_map'].to(net.device())

        # applying mask
        mask_x_gt = flow_target[:, :, :, 0].ge(-1) & flow_target[:, :, :, 0].le(1)
        mask_y_gt = flow_target[:, :, :, 1].ge(-1) & flow_target[:, :, :, 1].le(1)
        mask_xx_gt = mask_x_gt & mask_y_gt
        mask_gt = torch.cat((mask_xx_gt.unsqueeze(3), mask_xx_gt.unsqueeze(3)), dim=3)

        for i in range(bs):
            # unnormalize the flow: [-1;1] -> [0;239]
            flow_target[i] = (flow_target[i] + 1) * (img_size - 1) / (1 + 1)
            flow_est[i]    = (flow_est[i] + 1) * (img_size - 1) / (1 + 1)


        flow_target_x = flow_target[:, :, :, 0]
        flow_target_y = flow_target[:, :, :, 1]
        flow_est_x = flow_est[:, :, :, 0]
        flow_est_y = flow_est[:, :, :, 1]

        flow_target = torch.cat((flow_target_x[mask_gt[:, :, :, 0]].unsqueeze(1),
                                 flow_target_y[mask_gt[:, :, :, 1]].unsqueeze(1)), dim=1)
        flow_est = torch.cat((flow_est_x[mask_gt[:, :, :, 0]].unsqueeze(1),
                              flow_est_y[mask_gt[:, :, :, 1]].unsqueeze(1)), dim=1)

        # let's calculate EPE
        aepe = epe(flow_est, flow_target)
        aepe_array.append(aepe.item())
        n_registered_pxs += flow_target.shape[0]

    aepe_array = list(itertools.chain(*aepe_array))
    print(n_registered_pxs)
    return aepe_array


def calculate_pck_hpatches(net, val_loader, alpha=1, im_size=240):
    total_number_of_correspondences = 0
    total_correct_correspondences = 0

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for _, mini_batch in pbar:

        source_img = mini_batch['source_image'].to(net.device())
        target_img = mini_batch['target_image'].to(net.device())

        # network estimates
        estimates_grid = net(source_img, target_img)

        flow_est = estimates_grid[-1].transpose(1,2).transpose(2,3).to(net.device())
        flow_target = mini_batch['correspondence_map'].to(net.device())
        bs, ch_g, h_g, w_g = flow_target.shape

        # applying mask
        mask_x_gt = flow_target[:, :, :, 0].ge(-1) & flow_target[:, :, :, 0].le(1)
        mask_y_gt = flow_target[:, :, :, 1].ge(-1) & flow_target[:, :, :, 1].le(1)
        mask_xx_gt = mask_x_gt & mask_y_gt
        mask_gt = torch.cat((mask_xx_gt.unsqueeze(3), mask_xx_gt.unsqueeze(3)), dim=3)

        for i in range(bs):
            # unnormalize the flow: [-1;1] -> [0;239]
            flow_target[i] = (flow_target[i] + 1) * (img_size - 1) / (1 + 1)
            flow_est[i]    = (flow_est[i] + 1) * (img_size - 1) / (1 + 1)

        flow_target = flow_target.contiguous().view(1, bs * h_g * w_g * ch_g)
        flow_est    = flow_est.contiguous().view(1, bs * h_g * w_g * ch_g)
        # applying mask
        flow_target_m = flow_target[mask_gt.contiguous().view(1, bs * h_g * w_g * ch_g)]
        flow_est_m    = flow_est[mask_gt.contiguous().view(1, bs * h_g * w_g * ch_g)]

        total_number_of_correspondences += len(flow_target_m)
        total_correct_correspondences += correct_correspondences(flow_est_m, flow_target_m, alpha=alpha, im_size=im_size)

    pck = total_correct_correspondences / (total_number_of_correspondences + 1e-6)
    return pck