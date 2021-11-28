#
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#

import torch, sys, time
import scipy.io as scio
sys.path.append('..')

from dp_module import DPMergeModule as DPMerge


torch.manual_seed(0)
torch.cuda.manual_seed(0)

enable_load_matlab = True

if enable_load_matlab:
    obj = scio.loadmat('../../data/matlab.mat')
    RGB_img = torch.from_numpy(obj['RGB_img']).float().permute(2, 0, 1).unsqueeze(0)
    depth = torch.from_numpy(obj['k_size']).float().unsqueeze(0)
    img_left_gt = torch.from_numpy(obj['img_left']).float().permute(2, 0, 1).unsqueeze(0)
    img_right_gt = torch.from_numpy(obj['img_right']).float().permute(2, 0, 1).unsqueeze(0)
    count_left_gt = torch.from_numpy(obj['count_left']).float().unsqueeze(0)
    count_right_gt = torch.from_numpy(obj['count_right']).float().unsqueeze(0)
    b, c, h, w = RGB_img.shape

    count_left_gt[count_left_gt == 0] = 1
    count_right_gt[count_right_gt == 0] = 1
    img_left_avg_gt = img_left_gt / count_left_gt
    img_right_avg_gt = img_right_gt / count_right_gt

    DP_obj = DPMerge(enable_left=True, enable_right=True)

    torch.cuda.synchronize()
    time_start = time.time()

    # PyTorch module wrapper takes more time than CUDA
    # To check actual CUDA running time, go to ../python and run simulator_dp.py
    with torch.no_grad():
        img_left_avg_cu, img_right_avg_cu = DP_obj(RGB_img.cuda(), depth.cuda())

    duration = time.time() - time_start

    print('Check CUDA forward: left diff: {:.8f}, right diff: {:.8f}; left min: {:.8f}, max: {:8f}; time: {:.6f}s.' \
        .format((img_left_avg_cu.cpu() - img_left_avg_gt).abs().max().numpy(),
                (img_right_avg_cu.cpu() - img_right_avg_gt).abs().max().numpy(),
                img_left_avg_cu.cpu().min(), img_left_avg_cu.max(), duration))
else:
    # This is a basic demo for how to use this module,
    # to check if the depth gradient on CUDA is correct and 1000-time statistics,
    # go to ../python and run simulator_dp.py
    b, c, h, w, num_disps = 2, 3, 32, 64, 32
    RGB_img = torch.rand(size=(b, c, h, w), dtype=torch.float32, requires_grad=True).cuda()
    depth = torch.rand(size=(b, h, w), dtype=torch.float32, requires_grad=True).cuda() * num_disps

    DP_obj = DPMerge(enable_left=True, enable_right=True)
    img_left_avg_cu, img_right_avg_cu = DP_obj(RGB_img, depth)

    loss = (img_left_avg_cu * torch.rand(img_left_avg_cu.shape, device=img_left_avg_cu.device)).sum()
    loss += (img_right_avg_cu * torch.rand(img_right_avg_cu.shape, device=img_right_avg_cu.device)).sum()

    RGB_img.retain_grad()
    depth.retain_grad()
    loss.backward()
    dimage = RGB_img.grad
    ddepth = depth.grad

    print('RGB gradient shape: {}, min: {:.4f}, max: {:.4f}' \
          .format(dimage.shape, dimage.min(), dimage.max()))
    print('Depth gradient shape: {}, min: {:.4f}, max: {:.4f}' \
        .format(ddepth.shape, ddepth.min(), ddepth.max()))