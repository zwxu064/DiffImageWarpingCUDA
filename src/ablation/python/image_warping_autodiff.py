# -------------------------------------------------------------------------------------------------------------
# File: image_warping_autodiff.py
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

import torch

# Forward

# =============================================================================================================

def extrapolation(
    y,
    z,
    h,
    w,
    src_pixel
):
    img_list, count_list, y_int_list, z_int_list = [], [], [], []
    y = y.view(1)
    z = z.view(1)

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

# =============================================================================================================

def simdp_extrapol(
    RGB_img,
    depth
):
    ker_size = depth
    batch, c, h, w = RGB_img.shape
    device = RGB_img.device
    dtype = RGB_img.dtype

    img_left = torch.zeros(size=(batch, c, h, w), dtype=dtype).to(device)
    count_left = torch.zeros(size=(batch, h, w), dtype=dtype).to(device)
    img_right = torch.zeros(size=(batch, c, h, w), dtype=dtype).to(device)
    count_right = torch.zeros(size=(batch, h, w), dtype=dtype).to(device)

    for b in range(batch):
        for i in range(h):
            for j in range(w):
                y1 = torch.tensor([i], dtype=dtype, device=device)
                y2 = i + ker_size[b, i, j]
                z1 = torch.tensor([j], dtype=dtype, device=device)
                z2 = j + ker_size[b, i, j]

                # Synthesizing Left Image
                img_v_list, count_v_list, y_int_list, z_int_list = \
                    extrapolation(y1, z1, h, w, RGB_img[b, :, i, j])
                for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                    img_left[b, :, y_int, z_int] = img_left[b, :, y_int, z_int] + img_v
                    count_left[b, y_int, z_int] = count_left[b, y_int, z_int] + count_v

                img_v_list, count_v_list, y_int_list, z_int_list = \
                    extrapolation(y2, z1, h, w, RGB_img[b, :, i, j])
                for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                    img_left[b, :, y_int, z_int] = img_left[b, :, y_int, z_int] - img_v
                    count_left[b, y_int, z_int] = count_left[b, y_int, z_int] - count_v

                img_v_list, count_v_list, y_int_list, z_int_list = \
                    extrapolation(y1, z2, h, w, RGB_img[b, :, i, j])
                for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                    img_left[b, :, y_int, z_int] = img_left[b, :, y_int, z_int] - img_v
                    count_left[b, y_int, z_int] = count_left[b, y_int, z_int] - count_v

                img_v_list, count_v_list, y_int_list, z_int_list = \
                    extrapolation(y2, z2, h, w, RGB_img[b, :, i, j])
                for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                    img_left[b, :, y_int, z_int] = img_left[b, :, y_int, z_int] + img_v
                    count_left[b, y_int, z_int] = count_left[b, y_int, z_int] + count_v

                # Synthesizing Right Image
                z1 = j - ker_size[b, i, j]
                z2 = torch.tensor([j], dtype=dtype, device=device)

                img_v_list, count_v_list, y_int_list, z_int_list = \
                    extrapolation(y1, z1, h, w, RGB_img[b, :, i, j])
                for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                    img_right[b, :, y_int, z_int] = img_right[b, :, y_int, z_int] + img_v
                    count_right[b, y_int, z_int] = count_right[b, y_int, z_int] + count_v

                img_v_list, count_v_list, y_int_list, z_int_list = \
                    extrapolation(y2, z1, h, w, RGB_img[b, :, i, j])
                for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                    img_right[b, :, y_int, z_int] = img_right[b, :, y_int, z_int] - img_v
                    count_right[b, y_int, z_int] = count_right[b, y_int, z_int] - count_v

                img_v_list, count_v_list, y_int_list, z_int_list = \
                    extrapolation(y1, z2, h, w, RGB_img[b, :, i, j])
                for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                    img_right[b, :, y_int, z_int] = img_right[b, :, y_int, z_int] - img_v
                    count_right[b, y_int, z_int] = count_right[b, y_int, z_int] - count_v

                img_v_list, count_v_list, y_int_list, z_int_list = \
                    extrapolation(y2, z2, h, w, RGB_img[b, :, i, j])
                for img_v, count_v, y_int, z_int in zip(img_v_list, count_v_list, y_int_list, z_int_list):
                    img_right[b, :, y_int, z_int] = img_right[b, :, y_int, z_int] + img_v
                    count_right[b, y_int, z_int] = count_right[b, y_int, z_int] + count_v

    return img_left, img_right, count_left, count_right


# Backward

# =============================================================================================================

def extrapolation_back(
    y,
    z,
    h,
    w,
    src_pixel,
    is_positive,
    dimg, dcount,
    i=None,
    j=None
):
    dy, dz, dpixel_src = y.new_zeros(1), z.new_zeros(1), z.new_zeros(3)
    y = y.view(1)
    z = z.view(1)

    y_int_loop = torch.unique(torch.cat([torch.floor(y).detach(), torch.ceil(y).detach()], dim=0))
    z_int_loop = torch.unique(torch.cat([torch.floor(z).detach(), torch.ceil(z).detach()], dim=0))

    for y_int in y_int_loop:
        for z_int in z_int_loop:
            y_weight = 1 - abs(y_int - y)
            z_weight = 1 - abs(z_int - z)
            weight = y_weight * z_weight
            y_int_valid = torch.clamp(y_int, 0, h - 1).long()
            z_int_valid = torch.clamp(z_int, 0, w - 1).long()
            dimg_v = dimg[:, y_int_valid, z_int_valid]
            dcount_v = dcount[y_int_valid, z_int_valid]
            dweight = (dimg_v * src_pixel).sum(0) + dcount_v

            if not is_positive:
                dweight = -dweight
                weight = -weight

            dpixel_src += dimg_v * weight
            dy_weight = dweight * z_weight
            dz_weight = dweight * y_weight

            dy[y_int > y] += dy_weight[y_int > y]
            dy[y_int < y] -= dy_weight[y_int < y]
            dz[z_int > z] += dz_weight[z_int > z]
            dz[z_int < z] -= dz_weight[z_int < z]

    return dy, dz, dpixel_src

# =============================================================================================================

def simdp_extrapol_back(
    RGB_img,
    depth,
    dimg_left,
    dimg_right,
    dcount_left,
    dcount_right
):
    dtype = depth.dtype
    device = depth.device
    batch, h, w = depth.shape
    ddepth = torch.zeros(depth.shape, dtype=dtype, device=device)
    dRGB_img = torch.zeros(RGB_img.shape, dtype=dtype, device=device)
    ker_size = depth

    for b in range(batch):
        for i in range(h):
            for j in range(w):
                y1 = torch.tensor([i], dtype=dtype, device=device)
                y2 = i + ker_size[b, i, j]
                z1 = torch.tensor([j], dtype=dtype, device=device)
                z2 = j + ker_size[b, i, j]

                # Back Left Image
                dy1, dz1, dpixel_src = extrapolation_back(
                    y1, z1, h, w, RGB_img[b, :, i, j], True, dimg_left[b], dcount_left[b], i, j
                )
                dRGB_img[b, :, i, j] = dRGB_img[b, :, i, j] + dpixel_src

                dy2, dz1, dpixel_src = extrapolation_back(
                    y2, z1, h, w, RGB_img[b, :, i, j], False, dimg_left[b], dcount_left[b]
                )
                ddepth[b, i, j] = ddepth[b, i, j] + dy2
                dRGB_img[b, :, i, j] = dRGB_img[b, :, i, j] + dpixel_src

                dy1, dz2, dpixel_src = extrapolation_back(
                    y1, z2, h, w, RGB_img[b, :, i, j], False, dimg_left[b], dcount_left[b]
                )
                ddepth[b, i, j] = ddepth[b, i, j] + dz2
                dRGB_img[b, :, i, j] = dRGB_img[b, :, i, j] + dpixel_src

                dy2, dz2, dpixel_src = extrapolation_back(
                    y2, z2, h, w, RGB_img[b, :, i, j], True, dimg_left[b], dcount_left[b]
                )
                ddepth[b, i, j] = ddepth[b, i, j] + dy2 + dz2
                dRGB_img[b, :, i, j] = dRGB_img[b, :, i, j] + dpixel_src

                # Back Right Image
                z1 = j - ker_size[b, i, j]
                z2 = torch.tensor([j], dtype=dtype, device=device)

                dy1, dz1, dpixel_src = extrapolation_back(
                    y1, z1, h, w, RGB_img[b, :, i, j], True, dimg_right[b], dcount_right[b]
                )
                ddepth[b, i, j] = ddepth[b, i, j] - dz1
                dRGB_img[b, :, i, j] = dRGB_img[b, :, i, j] + dpixel_src

                dy2, dz1, dpixel_src = extrapolation_back(
                    y2, z1, h, w, RGB_img[b, :, i, j], False, dimg_right[b], dcount_right[b]
                )
                ddepth[b, i, j] = ddepth[b, i, j] + dy2 - dz1
                dRGB_img[b, :, i, j] = dRGB_img[b, :, i, j] + dpixel_src

                dy1, dz2, dpixel_src = extrapolation_back(
                    y1, z2, h, w, RGB_img[b, :, i, j], False, dimg_right[b], dcount_right[b]
                )
                dRGB_img[b, :, i, j] = dRGB_img[b, :, i, j] + dpixel_src

                dy2, dz2, dpixel_src = extrapolation_back(
                    y2, z2, h, w, RGB_img[b, :, i, j], True, dimg_right[b], dcount_right[b]
                )
                ddepth[b, i, j] = ddepth[b, i, j] + dy2
                dRGB_img[b, :, i, j] = dRGB_img[b, :, i, j] + dpixel_src

    return ddepth, dRGB_img