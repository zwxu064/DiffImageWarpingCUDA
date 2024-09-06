# -------------------------------------------------------------------------------------------------------------
# File: test_diff_image_warping.py
# Project: Differentiable Image Warping (CUDA)
# Contributors:
#     Zhiwei Xu <zwxu064@gmail.com>
# 
# Copyright (c) 2024 Zhiwei Xu
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------

import os
import torch, sys, time
import scipy.io as scio
import matplotlib.pyplot as plt

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from src.diff_image_warping import ImageWarpingLayer

# =============================================================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    root = f"{os.path.dirname(os.path.abspath(__file__))}/../.."
    data_dir = os.path.join(root, "data/single/matlab")
    result_dir = os.path.join(root, "results/single")
    data_index = 1

    load_matlab = True
    enable_save_results = True
    verbose = True
    os.makedirs(result_dir, exist_ok=True) if not os.path.exists(result_dir) else None

    if load_matlab:
        obj = scio.loadmat(os.path.join(data_dir, 'matlab_{:02d}.mat'.format(data_index)))
        RGB_img = torch.from_numpy(obj['RGB_img']).float().permute(2, 0, 1).unsqueeze(0)
        depth = torch.from_numpy(obj['k_size']).float().unsqueeze(0)
        img_left_gt = torch.from_numpy(obj['img_left']).float().permute(2, 0, 1).unsqueeze(0)
        img_right_gt = torch.from_numpy(obj['img_right']).float().permute(2, 0, 1).unsqueeze(0)
        count_left_gt = torch.from_numpy(obj['count_left']).float().unsqueeze(0)
        count_right_gt = torch.from_numpy(obj['count_right']).float().unsqueeze(0)
        img_left_final_gt = torch.from_numpy(obj['img_left_final']).float().permute(2, 0, 1).unsqueeze(0)
        img_right_final_gt = torch.from_numpy(obj['img_right_final']).float().permute(2, 0, 1).unsqueeze(0)

        img_left_final_gt = img_left_final_gt.clamp(min=0., max=1.)
        img_right_final_gt = img_right_final_gt.clamp(min=0., max=1.)

        b, c, h, w = RGB_img.shape
        print('$$$$$$', depth.shape)

        image_warping_layer = ImageWarpingLayer(enable_left=True, enable_right=True)

        torch.cuda.synchronize()
        time_start = time.time()

        # PyTorch module wrapper takes more time than CUDA
        # To check actual CUDA running time, go to ../python and run simulator_dp.py
        with torch.no_grad():
            results = image_warping_layer(RGB_img.cuda(), depth.cuda(), verbose=verbose)
            img_left_final_cu, img_right_final_cu = results[:2]

            if verbose:
                img_left_cu, img_right_cu, count_left_cu, count_right_cu = results[2:]

        duration = time.time() - time_start

        if enable_save_results:
            save_left_path = os.path.join(result_dir, 'left_syn_cu_{:02d}.png'.format(data_index))
            save_right_path = os.path.join(result_dir, 'right_syn_cu_{:02d}.png'.format(data_index))
            img_left_final_save = img_left_final_cu[0].permute(1, 2, 0).cpu().detach().numpy()
            img_right_final_save = img_right_final_cu[0].permute(1, 2, 0).cpu().detach().numpy()

            plt.figure()
            plt.subplot(221)
            plt.title('Left-MatLab')
            plt.imshow(img_left_final_gt[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.subplot(222)
            plt.title('Right-MatLab')
            plt.imshow(img_right_final_gt[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.subplot(223)
            plt.title('Left-CUDA')
            plt.imshow(img_left_final_save)
            plt.subplot(224)
            plt.title('Right-CUDA')
            plt.imshow(img_right_final_save)
            plt.show()

            plt.imsave(save_left_path, img_left_final_save)
            plt.imsave(save_right_path, img_right_final_save)

        if verbose:
            print(
                f"(Should have small difference)\n" \
                f"    Check CUDA forward RAW: " \
                f"left diff: {(img_left_cu.cpu() - img_left_gt).abs().max().numpy():.8f}" \
                f", right diff: {(img_right_cu.cpu() - img_right_gt).abs().max().numpy():.8f}" \
                f"; left count diff: {(count_left_cu.cpu() - count_left_gt).abs().max().numpy():.8f}" \
                f", right count diff: {(count_right_cu.cpu() - count_right_gt).abs().max().numpy():.8f}" \
                f"; time: {duration:.6f}s."
            )

        print(
            f"(Huge difference is OK due to different MatLab function)\n" \
            f"    Check CUDA forward FINAL: " \
            f"left diff: {(img_left_final_cu.cpu() - img_left_final_gt).abs().max().numpy():.8f}" \
            f", right diff: {(img_right_final_cu.cpu() - img_right_final_gt).abs().max().numpy():.8f}" \
            f"; left min: {img_left_final_cu.cpu().min():.8f}" \
            f", max: {img_left_final_cu.max():8f}" \
            f"; time: {duration:.6f}s."
        )
    else:
        # This is a basic demo for how to use this module,
        # to check if the depth gradient on CUDA is correct and 1000-time statistics,
        # go to ../python and run simulator_dp.py
        b, c, h, w, num_disps = 2, 3, 32, 64, 32
        RGB_img = torch.rand(size=(b, c, h, w), dtype=torch.float32, requires_grad=True).cuda()
        depth = torch.rand(size=(b, h, w), dtype=torch.float32, requires_grad=True).cuda() * num_disps

        image_warping_layer = ImageWarpingLayer(enable_left=True, enable_right=True)
        img_left_avg_cu, img_right_avg_cu = image_warping_layer(RGB_img, depth)

        loss = (img_left_avg_cu * torch.rand(img_left_avg_cu.shape, device=img_left_avg_cu.device)).sum()
        loss += (img_right_avg_cu * torch.rand(img_right_avg_cu.shape, device=img_right_avg_cu.device)).sum()

        RGB_img.retain_grad()
        depth.retain_grad()
        loss.backward()
        dimage = RGB_img.grad
        ddepth = depth.grad

        print(f'RGB gradient shape: {dimage.shape}, min: {dimage.min():.4f}, max: {dimage.max():.4f}')
        print(f'Depth gradient shape: {ddepth.shape}, min: {ddepth.min():.4f}, max: {ddepth.max():.4f}')