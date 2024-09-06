# -------------------------------------------------------------------------------------------------------------
# File: diff_image_warping.py
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

import torch, sys, os

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from src.cuda.lib_diffimagewarping import DiffImageWarping

# =============================================================================================================

class ImageWarpingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        image,
        depth,
        enable_left=False,
        enable_right=False
    ):
        b, c, h, w = image.shape

        image = image.permute(0, 2, 3, 1).contiguous()
        depth = depth.contiguous()

        if enable_left:
            count_left = image.new_zeros(size=(b, h, w))
            img_left = image.new_zeros(size=(b, h, w, c))
        else:
            count_left = torch.empty(0, dtype=image.dtype, device=image.device)
            img_left = torch.empty(0, dtype=image.dtype, device=image.device)

        if enable_right:
            count_right = image.new_zeros(size=(b, h, w))
            img_right = image.new_zeros(size=(b, h, w, c))
        else:
            count_right = torch.empty(0, dtype=image.dtype, device=image.device)
            img_right = torch.empty(0, dtype=image.dtype, device=image.device)

        # assert count_left.is_contiguous() and count_right.is_contiguous()
        # assert img_left.is_contiguous() and img_right.is_contiguous()

        DiffImageWarping.ImageWarping(
            image,
            depth,
            count_left,
            count_right,
            img_left,
            img_right
        )

        ctx.intermediate_results = image, depth
        img_left = img_left.permute(0, 3, 1, 2)
        img_right = img_right.permute(0, 3, 1, 2)

        return count_left, count_right, img_left, img_right

    @staticmethod
    def backward(
        ctx,
        dcount_left,
        dcount_right,
        dimg_left,
        dimg_right
    ):
        image, depth = ctx.intermediate_results
        b, h, w, c = image.shape
        dimage = image.new_zeros(size=(b, h, w, c)).contiguous()
        ddepth = image.new_zeros(size=(b, h, w)).contiguous()

        # assert dcount_left.is_contiguous() and dcount_right.is_contiguous()
        # assert dimg_left.is_contiguous() and dimg_right.is_contiguous()

        DiffImageWarping.ImageWarpingBack(
            image,
            depth,
            dcount_left,
            dcount_right,
            dimg_left,
            dimg_right,
            ddepth,
            dimage
        )

        dimage = dimage.permute(0, 3, 1, 2)

        return dimage, ddepth, None, None

# =============================================================================================================

class ImageWarpingLayer(torch.nn.Module):
    def __init__(
        self,
        enable_left=False,
        enable_right=False
    ):
        super(ImageWarpingLayer, self).__init__()
        self.enable_left = enable_left
        self.enable_right = enable_right

    def forward(
        self,
        image,
        depth,
        verbose=False
    ):
        count_left, count_right, img_left, img_right = ImageWarpingFunction.apply(
            image,
            depth,
            self.enable_left,
            self.enable_right
        )

        # Accumulated images.
        count_left_final = count_left.cumsum(dim=1).cumsum(dim=2)
        count_right_final = count_right.cumsum(dim=1).cumsum(dim=2)
        count_left_final[count_left_final <= 1.0] = 1.0
        count_right_final[count_right_final <= 1.0] = 1.0

        img_left_avg = img_left.cumsum(dim=2).cumsum(dim=3) / count_left_final.unsqueeze(1)
        img_right_avg = img_right.cumsum(dim=2).cumsum(dim=3) / count_right_final.unsqueeze(1)

        img_left_avg = img_left_avg.clamp(min=0., max=1.)
        img_right_avg = img_right_avg.clamp(min=0., max=1.)

        if verbose:
            return img_left_avg, img_right_avg, img_left, img_right, count_left, count_right
        else:
            return img_left_avg, img_right_avg