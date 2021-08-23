#
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#

import torch, sys
import torch.nn as nn
sys.path.append('..')

from cuda.lib_dualpixel import DualPixel


class DPMergeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, depth, enable_left=False, enable_right=False):
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

        DualPixel.DepthMerge(image,
                             depth,
                             count_left,
                             count_right,
                             img_left,
                             img_right)

        ctx.intermediate_results = image, depth
        img_left = img_left.permute(0, 3, 1, 2)
        img_right = img_right.permute(0, 3, 1, 2)

        return count_left, count_right, img_left, img_right

    @staticmethod
    def backward(ctx, dcount_left, dcount_right, dimg_left, dimg_right):
        image, depth = ctx.intermediate_results
        b, h, w, _ = image.shape
        ddepth = image.new_zeros(size=(b, h, w)).contiguous()

        # assert dcount_left.is_contiguous() and dcount_right.is_contiguous()
        # assert dimg_left.is_contiguous() and dimg_right.is_contiguous()

        DualPixel.DepthMergeBack(image,
                                 depth,
                                 dcount_left,
                                 dcount_right,
                                 dimg_left,
                                 dimg_right,
                                 ddepth)

        return None, ddepth, None, None


class DPMergeModule(torch.nn.Module):
    def __init__(self, enable_left=False, enable_right=False):
        super(DPMergeModule, self).__init__()
        self.enable_left = enable_left
        self.enable_right = enable_right

    def forward(self, image, depth):
        count_left, count_right, img_left, img_right = \
            DPMergeFunction.apply(image,
                                  depth,
                                  self.enable_left,
                                  self.enable_right)

        count_left[count_left == 0] = 1
        count_right[count_right == 0] = 1
        img_left_avg = img_left / count_left.unsqueeze(1)
        img_right_avg = img_right / count_right.unsqueeze(1)

        return img_left_avg, img_right_avg
