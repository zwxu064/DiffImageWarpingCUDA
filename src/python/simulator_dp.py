#
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#

import torch
import numpy as np
import time
import cv2
import scipy.io as scio
from dp_autodiff import simdp_extrapol, simdp_extrapol_back
import sys
sys.path.append('..')
from cuda.lib_dualpixel import DualPixel


def simdp(RGB_img, depth, image_left=None, image_right=None):
    cpsize = 5
    # depth_scaled = depth*5.0/(depth.max())
    ker_size = depth

    S = np.shape(depth) # (8, 192, 192)
    b = S[0]
    h = S[1]
    w = S[2]
    device = RGB_img.device

    img_left = torch.zeros([b,3,h,w]).to(device)  # np.zeros([b,3,h,w]), dtype=torch.long
    count_left = torch.zeros([b,1,h,w]).to(device)  # np.zeros([b,1,h,w])
    img_right = torch.zeros([b,3,h,w]).to(device)  # np.zeros([b,3,h,w])
    count_right = torch.zeros([b,1,h,w]).to(device)  # np.zeros([b,1,h,w])

    # Determining the projected regions on the image plane for every pixel position of the input image/scene
    for i in range(h):
        for j in range(w):
            # if ker_size[:,i,j]<1:
            #     continue

            y1 = (i - ker_size[:,i,j]).floor()
            y2 = (i + ker_size[:,i,j]).floor()
            z1 = torch.tensor([j] * b, dtype=ker_size.dtype, device=device)
            z2 = (j + ker_size[:,i,j]).floor()

            if False:
                if False:
                    if y1 == y2:
                        y2 = y2 + 1

                    if z1 == z2:
                        z2 = z2 + 1
                else:
                    y2[y1 == y2] += 1
                    z2[z1 == z2] += 1

            y1 = y1.clamp(0, h-1).cpu().numpy().astype(np.long)
            y2 = y2.clamp(0, h-1).cpu().numpy().astype(np.long)
            z1 = z1.clamp(0, w-1).cpu().numpy().astype(np.long)
            z2 = z2.clamp(0, w-1).cpu().numpy().astype(np.long)

            # Synthesizing Left Image
            img_left[:,:,y1,z1] = img_left[:,:,y1,z1] + RGB_img[:,:,i,j].unsqueeze(2)
            img_left[:,:,y2,z1] = img_left[:,:,y2,z1] - RGB_img[:,:,i,j].unsqueeze(2)
            img_left[:,:,y1,z2] = img_left[:,:,y1,z2] - RGB_img[:,:,i,j].unsqueeze(2)
            img_left[:,:,y2,z2] = img_left[:,:,y2,z2] + RGB_img[:,:,i,j].unsqueeze(2)

            count_left[:,:,y1,z1] = count_left[:,:,y1,z1] + 1
            count_left[:,:,y2,z1] = count_left[:,:,y2,z1] - 1
            count_left[:,:,y1,z2] = count_left[:,:,y1,z2] - 1
            count_left[:,:,y2,z2] = count_left[:,:,y2,z2] + 1

            z1 = torch.tensor([j] * b, dtype=ker_size.dtype, device=device)
            z2 = (j - ker_size[:,i,j]).floor()

            if False:
                if False:
                    if z1 == z2:
                        z2 = z2 - 1
                else:
                    z2[z1 == z2] -= 1

            z1 = z1.clamp(0, w-1).cpu().numpy().astype(np.long)
            z2 = z2.clamp(0, w-1).cpu().numpy().astype(np.long)

            # Synthesizing Right Image
            img_right[:,:,y1,z1] = img_right[:,:,y1,z1] + RGB_img[:,:,i,j].unsqueeze(2)
            img_right[:,:,y2,z1] = img_right[:,:,y2,z1] - RGB_img[:,:,i,j].unsqueeze(2)
            img_right[:,:,y1,z2] = img_right[:,:,y1,z2] - RGB_img[:,:,i,j].unsqueeze(2)
            img_right[:,:,y2,z2] = img_right[:,:,y2,z2] + RGB_img[:,:,i,j].unsqueeze(2)

            count_right[:,:,y1,z1] = count_right[:,:,y1,z1] + 1
            count_right[:,:,y2,z1] = count_right[:,:,y2,z1] - 1
            count_right[:,:,y1,z2] = count_right[:,:,y1,z2] - 1
            count_right[:,:,y2,z2] = count_right[:,:,y2,z2] + 1

    integral_image = (cv2.integral(255 * img_left.squeeze().cpu().detach().numpy().transpose(1, 2, 0)))
    integral_count = (cv2.integral(255 * count_left.squeeze().cpu().detach().numpy()))
    integral_image = torch.from_numpy(integral_image).permute(2, 0, 1).float().unsqueeze(0) / 255
    integral_count = torch.from_numpy(integral_count).float().unsqueeze(0).unsqueeze(1) / 255
    integral_count = integral_count  # .clamp(1e-4, 1)
    im_left = (integral_image / integral_count).clamp(-0.5, 0.5)

    integral_image = (cv2.integral(255 * img_right.squeeze().cpu().detach().numpy().transpose(1, 2, 0)))
    integral_count = (cv2.integral(255 * count_right.squeeze().cpu().detach().numpy()))
    integral_image = torch.from_numpy(integral_image).permute(2, 0, 1).float().unsqueeze(0) / 255
    integral_count = torch.from_numpy(integral_count).float().unsqueeze(0).unsqueeze(1) / 255
    integral_count = integral_count  # .clamp(1e-4, 1)

    im_right = (integral_image / integral_count).clamp(-0.5, 0.5)

    return im_left[:,:,:-1,:-1].cuda(), im_right[:,:,:-1,:-1].cuda(), \
           img_left, img_right, count_left, count_right


if __name__ == '__main__':
    torch.manual_seed(0)

    enable_extrapol = True
    enable_load_matlab = True  # enable for real data; otherwise, random data
    enable_cuda = True
    enable_gradient = True
    enable_pytorch_manual = True  # this will be super slow
    enable_cuda_implement = True

    enable_gradient = False if (not enable_pytorch_manual) else enable_gradient
    enable_gradient = False if (enable_load_matlab) else enable_gradient

    if enable_cuda:
        torch.cuda.manual_seed(0)

    # ==== Check forward propagation
    if enable_load_matlab:
        obj = scio.loadmat('../../data/matlab.mat')
        RGB_img = torch.from_numpy(obj['RGB_img']).float().permute(2, 0, 1).unsqueeze(0)
        depth = torch.from_numpy(obj['k_size']).float().unsqueeze(0)
        img_left_gt = torch.from_numpy(obj['img_left']).float().permute(2, 0, 1).unsqueeze(0)
        img_right_gt = torch.from_numpy(obj['img_right']).float().permute(2, 0, 1).unsqueeze(0)
        count_left_gt = torch.from_numpy(obj['count_left']).float().unsqueeze(0)
        count_right_gt = torch.from_numpy(obj['count_right']).float().unsqueeze(0)
        depth.requires_grad = True if enable_gradient else False
        depth_in = depth / 2
        b, c, h, w = RGB_img.shape

        if enable_cuda:
            RGB_img = RGB_img.cuda()
            depth_in = depth_in.cuda()

        print('GT:',
              img_left_gt.abs().sum().numpy(),
              img_right_gt.abs().sum().numpy(),
              count_left_gt.abs().sum().numpy(),
              count_right_gt.abs().sum().numpy())
    else:
        b, c, h, w, num_disps = 8, 3, 32, 64, 32
        RGB_img = torch.rand(size=(b, c, h, w), dtype=torch.float32)
        depth = torch.rand(size=(b, h, w), dtype=torch.float32, requires_grad=enable_gradient) * num_disps
        depth_in = depth

    if enable_extrapol:
        img_left, img_right, count_left, count_right = None, None, None, None

        if enable_pytorch_manual:
            time_start = time.time()
            img_left, img_right, count_left, count_right = simdp_extrapol(RGB_img, depth_in)
            duration = time.time() - time_start

            print('Max self forward: img left: {:.8f}, right: {:.8f}; '
                  'count left: {:.8f}, right: {:.8f}; time: {:.6f}s'.format(
                img_left.cpu().detach().abs().max().numpy(),
                img_right.cpu().detach().abs().max().numpy(),
                count_left.cpu().detach().abs().max().numpy(),
                count_right.cpu().detach().abs().max().numpy(),
                duration))

        if enable_cuda_implement:
            device = RGB_img.device
            RGB_img_permute = RGB_img.permute(0, 2, 3, 1).contiguous()
            depth_in = depth_in.contiguous()
            RGB_img_permute_cu = RGB_img_permute.cuda()
            depth_in_cu = depth_in.cuda()
            duration_cuda = 0

            for idx in range(1000):
                count_left_cu = torch.zeros(b, h, w, dtype=torch.float32, device='cuda')
                count_right_cu = torch.zeros(b, h, w, dtype=torch.float32, device='cuda')
                img_left_cu = RGB_img.new_zeros(size=(b, h, w, c), device='cuda')
                img_right_cu = RGB_img.new_zeros(size=(b, h, w, c), device='cuda')
                assert count_left_cu.is_contiguous() and count_right_cu.is_contiguous()
                assert img_left_cu.is_contiguous() and img_right_cu.is_contiguous()

                torch.cuda.synchronize()
                time_start = time.time()
                DualPixel.DepthMerge(RGB_img_permute_cu,
                                     depth_in_cu,
                                     count_left_cu,
                                     count_right_cu,
                                     img_left_cu,
                                     img_right_cu)
                torch.cuda.synchronize()
                duration_cuda += time.time() - time_start

            duration_cuda /= 1000

            img_left_cu = img_left_cu.permute(0, 3, 1, 2).contiguous()
            img_right_cu = img_right_cu.permute(0, 3, 1, 2).contiguous()

            if enable_load_matlab:
                print('Check CUDA forward: diff left img: {:.8f}, count: {:.8f}; '
                      'right img: {:.8f}, count: {:.8f}; max left img: {:.8f}, max right img: {:.8f}; '
                      'time: {:.6f}s'.format(
                      (img_left_cu.cpu() - img_left_gt).abs().max().numpy(),
                      (count_left_cu.cpu() - count_left_gt).abs().max().numpy(),
                      (img_right_cu.cpu() - img_right_gt).abs().max().numpy(),
                      (count_right_cu.cpu() - count_right_gt).abs().max().numpy(),
                      img_left_cu.cpu().abs().max(), img_right_cu.cpu().abs().max(),
                      duration_cuda))
            elif img_left is not None:
                print('Check CUDA forward: diff left img: {:.8f}, count: {:.8f}; '
                      'right img: {:.8f}, count: {:.8f}; time: {:.6f}s'.format(
                      (img_left_cu.cpu() - img_left.detach()).abs().max().numpy(),
                      (count_left_cu.cpu() - count_left.detach()).abs().max().numpy(),
                      (img_right_cu.cpu() - img_right.detach()).abs().max().numpy(),
                      (count_right_cu.cpu() - count_right.detach()).abs().max().numpy(),
                      duration_cuda))
    else:
        time_start = time.time()
        _, _, img_left, img_right, count_left, count_right = simdp(RGB_img, depth_in)
        duration = time.time() - time_start

    if (img_left is not None) and enable_load_matlab:
        print('Check PyTorch forward: diff left img: {:.8f}, count: {:.8f}; '
              'right img: {:.8f}, count: {:.8f}; time: {:.6f}s'.format(
              (img_left.cpu().detach() - img_left_gt).abs().max().numpy(),
              (count_left.cpu().detach() - count_left_gt).abs().max().numpy(),
              (img_right.cpu().detach() - img_right_gt).abs().max().numpy(),
              (count_right.cpu().detach() - count_right_gt).abs().max().numpy(),
              duration))

    # ==== Check backward propagation
    if enable_gradient and (img_left is not None):
        if True:
            count_left_clone = count_left.clone()
            count_right_clone = count_right.clone()
            count_left_clone[count_left_clone == 0] = 1
            count_right_clone[count_right_clone == 0] = 1
        else:  # this will change count_left gradients, not much difference
            count_left[count_left == 0] = 1
            count_right[count_right == 0] = 1

        if True:
            loss = (img_left / count_left_clone.unsqueeze(1) + img_right / count_right_clone.unsqueeze(1)).mean()
        else:  # for debugging
            loss = 0
            loss += (img_left * torch.rand(img_left.shape)).sum()
            loss += (img_right * torch.rand(img_right.shape)).sum()
            loss += (count_left_clone * torch.rand(count_left_clone.shape)).sum()
            loss += (count_right_clone * torch.rand(count_right_clone.shape)).sum()
            loss = 1000 * loss

        depth.retain_grad()
        img_left.retain_grad()
        img_right.retain_grad()
        count_left.retain_grad()
        count_right.retain_grad()

        # Get autodiff
        time_start = time.time()
        loss.backward()
        duration = time.time() - time_start

        print('Autodiff backward: min: {:.8f}, max: {:.8f}; time: {:.6f}s'.format(
              depth.grad.min().detach().numpy(),
              depth.grad.max().detach().numpy(),
              duration))

        # Check manual back
        dimg_left = img_left.grad if (img_left.grad is not None) else torch.zeros_like(img_left)
        dimg_right = img_right.grad if (img_right.grad is not None) else torch.zeros_like(img_right)
        dcount_left = count_left.grad if (count_left.grad is not None) else torch.zeros_like(count_left)
        dcount_right = count_right.grad if (count_right.grad is not None) else torch.zeros_like(count_right)

        time_start = time.time()
        ddepth_manual = simdp_extrapol_back(RGB_img,
                                            depth_in,
                                            dimg_left,
                                            dimg_right,
                                            dcount_left,
                                            dcount_right)
        duration = time.time() - time_start

        print('Check manual backward: diff: {:.8f}; '
              'min: {:.8f}, max: {:.8f}; time: {:.6f}s'.format(
              (depth.grad - ddepth_manual).abs().max().detach().numpy(),
              ddepth_manual.min().detach().numpy(),
              ddepth_manual.max().detach().numpy(),
              duration))

        # Check CUDA back
        if enable_cuda_implement:
            dcount_left_cu = dcount_left.cuda()
            dcount_right_cu = dcount_right.cuda()
            dimg_left_cu = dimg_left.permute(0, 2, 3, 1).cuda()
            dimg_right_cu = dimg_right.permute(0, 2, 3, 1).cuda()

            assert RGB_img_permute.is_contiguous() and depth_in.is_contiguous()
            assert dcount_left.is_contiguous() and dcount_right.is_contiguous()
            assert dimg_left.is_contiguous() and dimg_right.is_contiguous()

            duration = 0
            for idx in range(1000):
                ddepth_cu = torch.zeros(b, h, w, dtype=torch.float32, device='cuda').contiguous()
                torch.cuda.synchronize()
                time_start = time.time()
                DualPixel.DepthMergeBack(RGB_img_permute_cu,
                                         depth_in_cu,
                                         dcount_left_cu,
                                         dcount_right_cu,
                                         dimg_left_cu,
                                         dimg_right_cu,
                                         ddepth_cu)
                torch.cuda.synchronize()
                duration += time.time() - time_start
            duration /= 1000

            ddepth_cu = ddepth_cu.contiguous()

            print('Check CUDA backward: diff: {:.8f}; '
                  'min: {:.8f}, max: {:.8f}; time: {:.6f}s'.format(
                  (depth.grad - ddepth_cu.cpu()).abs().max().numpy(),
                  ddepth_cu.cpu().min().numpy(),
                  ddepth_cu.cpu().max().numpy(),
                  duration))
