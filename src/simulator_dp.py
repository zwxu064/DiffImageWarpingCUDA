import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py, time
import matplotlib.pyplot as plt
import cv2
from torch.autograd import Variable
# from multiprocessing import Process, Manager
import operator
from torch.multiprocessing import Pool, Process, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import scipy.io as scio


# Zhiwei
def extrapolation(y, z, h, w, src_pixel):
    img_list, count_list, y_int_list, z_int_list = [], [], [], []

    if True:
        y_int_loop = torch.unique(torch.cat([torch.floor(y).detach(), torch.ceil(y).detach()], dim=0))
        z_int_loop = torch.unique(torch.cat([torch.floor(z).detach(), torch.ceil(z).detach()], dim=0))
    else:
        # Note: torch.round() to the even intergral on both CPU and GPU,
        # so use floor to compare with CUDA and MatLab
        y_int_loop = torch.unique(torch.cat([torch.floor(y).detach()], dim=0))
        z_int_loop = torch.unique(torch.cat([torch.floor(z).detach()], dim=0))
        y = torch.floor(y)
        z = torch.floor(z)

    for y_int in y_int_loop:
        for z_int in z_int_loop:
            y_weight = 1 - abs(y_int - y)
            z_weight = 1 - abs(z_int - z)
            weight = y_weight * z_weight

            # Note: rename y_int and z_int
            # Note: this clamp assigns values to edges for final accumulated images
            y_int_valid = torch.clamp(y_int, 0, h - 1)
            z_int_valid = torch.clamp(z_int, 0, w - 1)

            img_v = weight * src_pixel
            count_v = weight

            img_list.append(img_v)
            count_list.append(count_v)
            y_int_list.append(y_int_valid.long())
            z_int_list.append(z_int_valid.long())

    return img_list, count_list, y_int_list, z_int_list


# Zhiwei
def simdp_extrapol(RGB_img, depth):
    ker_size = depth
    S = np.shape(depth)
    b = S[0]
    h = S[1]
    w = S[2]
    device = RGB_img.device
    dtype = RGB_img.dtype

    img_left = torch.zeros([b, 3, h, w], dtype=dtype).to(device)
    count_left = torch.zeros([b, 1, h, w], dtype=dtype).to(device)
    img_right = torch.zeros([b, 3, h, w], dtype=dtype).to(device)
    count_right = torch.zeros([b, 1, h, w], dtype=dtype).to(device)

    for i in range(h):
        for j in range(w):
            y1 = i - ker_size[:, i, j]
            y2 = i + ker_size[:, i, j]
            z1 = torch.tensor([j] * b, dtype=dtype, device=device)
            z2 = j + ker_size[:, i, j]

            # Synthesizing Left Image
            img_v_list, count_v_list, y_int_list, z_int_list = \
                extrapolation(y1, z1, h, w, RGB_img[:, :, i, j])
            for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                img_left[:, :, y_int, z_int] = img_left[:, :, y_int, z_int] + img_v
                count_left[:, :, y_int, z_int] = count_left[:, :, y_int, z_int] + count_v

            img_v_list, count_v_list, y_int_list, z_int_list = \
                extrapolation(y2, z1, h, w, RGB_img[:, :, i, j])
            for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                img_left[:, :, y_int, z_int] = img_left[:, :, y_int, z_int] - img_v
                count_left[:, :, y_int, z_int] = count_left[:, :, y_int, z_int] - count_v

            img_v_list, count_v_list, y_int_list, z_int_list = \
                extrapolation(y1, z2, h, w, RGB_img[:, :, i, j])
            for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                img_left[:, :, y_int, z_int] = img_left[:, :, y_int, z_int] - img_v
                count_left[:, :, y_int, z_int] = count_left[:, :, y_int, z_int] - count_v

            img_v_list, count_v_list, y_int_list, z_int_list = \
                extrapolation(y2, z2, h, w, RGB_img[:, :, i, j])
            for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                img_left[:, :, y_int, z_int] = img_left[:, :, y_int, z_int] + img_v
                count_left[:, :, y_int, z_int] = count_left[:, :, y_int, z_int] + count_v

            # Synthesizing Right Image
            z1 = torch.tensor([j] * b, dtype=dtype, device=device)
            z2 = j - ker_size[:, i, j]

            img_v_list, count_v_list, y_int_list, z_int_list = \
                extrapolation(y1, z1, h, w, RGB_img[:, :, i, j])
            for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                img_right[:, :, y_int, z_int] = img_right[:, :, y_int, z_int] + img_v
                count_right[:, :, y_int, z_int] = count_right[:, :, y_int, z_int] + count_v

            img_v_list, count_v_list, y_int_list, z_int_list = \
                extrapolation(y2, z1, h, w, RGB_img[:, :, i, j])
            for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                img_right[:, :, y_int, z_int] = img_right[:, :, y_int, z_int] - img_v
                count_right[:, :, y_int, z_int] = count_right[:, :, y_int, z_int] - count_v

            img_v_list, count_v_list, y_int_list, z_int_list = \
                extrapolation(y1, z2, h, w, RGB_img[:, :, i, j])
            for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                img_right[:, :, y_int, z_int] = img_right[:, :, y_int, z_int] - img_v
                count_right[:, :, y_int, z_int] = count_right[:, :, y_int, z_int] - count_v

            img_v_list, count_v_list, y_int_list, z_int_list = \
                extrapolation(y2, z2, h, w, RGB_img[:, :, i, j])
            for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                img_right[:, :, y_int, z_int] = img_right[:, :, y_int, z_int] + img_v
                count_right[:, :, y_int, z_int] = count_right[:, :, y_int, z_int] + count_v

    return img_left, img_right, count_left, count_right


# Liyuan
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
    if True:  # some errors
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        enable_extrapol = True
        enable_load_matlab = True
        enable_cuda = False
        enable_gradient = False

        if enable_load_matlab:
            obj = scio.loadmat('data/matlab.mat')
            RGB_img = torch.from_numpy(obj['RGB_img']).float().permute(2, 0, 1).unsqueeze(0)
            depth = torch.from_numpy(obj['k_size']).float().unsqueeze(0)
            img_left_gt = torch.from_numpy(obj['img_left']).float().permute(2, 0, 1).unsqueeze(0)
            img_right_gt = torch.from_numpy(obj['img_right']).float().permute(2, 0, 1).unsqueeze(0)
            count_left_gt = torch.from_numpy(obj['count_left']).float().unsqueeze(0)
            count_right_gt = torch.from_numpy(obj['count_right']).float().unsqueeze(0)
            depth.requires_grad = True if enable_gradient else False
            depth_in = depth / 2

            if enable_cuda:
                RGB_img = RGB_img.cuda()
                depth_in = depth_in.cuda()

            print('GT:', img_left_gt.abs().sum(), img_right_gt.abs().sum(),
                  count_left_gt.abs().sum(), count_right_gt.abs().sum())
        else:
            b, c, h, w, num_disps = 1, 3, 32, 64, 16
            RGB_img = torch.randn(size=(b, c, h, w), dtype=torch.float32)
            depth = torch.randn(size=(b, h, w), dtype=torch.float32, requires_grad=enable_gradient)
            depth_in = depth

        time_start = time.time()

        if enable_extrapol:
            img_left, img_right, count_left, count_right = simdp_extrapol(RGB_img, depth_in)
        else:
            _, _, img_left, img_right, count_left, count_right = simdp(RGB_img, depth_in)

        duration = time.time() - time_start

        if enable_load_matlab:
            print('Check:', (img_left.cpu() - img_left_gt).abs().max(),
                  (count_left.cpu() - count_left_gt).abs().max(),
                  (img_right.cpu() - img_right_gt).abs().max(),
                  (count_right.cpu() - count_right_gt).abs().max(),
                  duration)
        else:
            print('Check:', img_left.abs().sum(), img_right.abs().sum(),
                  count_left.abs().sum(), count_right.abs().sum(),
                  duration)

        if enable_gradient:
            loss = (img_left + img_right).sum()
            depth.retain_grad()
            time_start = time.time()
            loss.backward()
            duration = time.time() - time_start
            print('Gradient:', depth.grad.shape, depth.grad.min(), depth.grad.max(), duration)

    if False:
        image = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32).view(1, 1, 2, 4)
        depth = torch.tensor([[[1./4, 1./4, 1./4, 1./4], [1./4, 1./4, 1./4, 1./4]]],
                             dtype=torch.float32, requires_grad=True)

        x_base = torch.linspace(0, 1, 4).repeat(1, 2, 1)
        y_base = torch.linspace(0, 1, 2).repeat(1, 4, 1).transpose(2, 1)
        flow_field = torch.stack((x_base + depth, y_base), dim=3)
        output = torch.nn.functional.grid_sample(image, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')
        loss = output.sum()
        depth.retain_grad()
        loss.backward()
        print(depth.shape, output.flatten(), depth.grad.flatten())  # this depth.grad is the same as my manual values
