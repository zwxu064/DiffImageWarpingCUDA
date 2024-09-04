#
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#

import torch, time

from CUDA.lib_dualpixel import DualPixel
from scipy import io as scio


torch.manual_seed(2021)
torch.cuda.manual_seed_all(2021)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    device = 'cuda'
    enable_dump_matlab = True
    loops = 10001

    # Data setup
    if not enable_dump_matlab:
        batch, channel, h, w = 1, 3, 1, 3
        image = torch.tensor(
            [[1, 4, 7, 2, 5, 8, 3, 6, 9]],
            dtype=torch.float32,
            device=device
        ).view(batch, channel, h, w)
        depth = torch.tensor([[2, 2, 2]], dtype=torch.float32, device=device).view(batch, h, w)

        left_img_count = torch.zeros(batch, h, w, dtype=torch.int32, device=device)
        right_img_count = torch.zeros(batch, h, w, dtype=torch.int32, device=device)
        left_img_stack = image.new_zeros(size=(batch, channel, h, w))
        right_img_stack = image.new_zeros(size=(batch, channel, h, w))

        image = image.permute(0, 2, 3, 1)
        left_img_stack = left_img_stack.permute(0, 2, 3, 1)
        right_img_stack = right_img_stack.permute(0, 2, 3, 1)

        DualPixel.DepthMerge(
            image,
            depth,
            left_img_count,
            right_img_count,
            left_img_stack,
            right_img_stack
        )
    else:
        # Load from MatLab dumped data, which was generated in "simulator_dp.m"
        data = scio.loadmat('../data/matlab_dump.mat')
        image = torch.from_numpy(data['RGB_img_s']).float().cuda() * 1
        depth = torch.from_numpy(data['k_size']).float().cuda() * 1
        left_img_count_gt = torch.from_numpy(data['count_left_s']).int().cuda() * 1
        right_img_count_gt = torch.from_numpy(data['count_right_s']).int().cuda() * 1
        left_img_gt = torch.from_numpy(data['img_left_s']).float().cuda() * 1
        right_img_gt = torch.from_numpy(data['img_right_s']).float().cuda() * 1

        batch = 1
        h, w, channel = image.shape
        image = image.view(batch, h, w, channel).contiguous()
        depth = depth.view(batch, h, w).contiguous()
        left_img_count_gt = left_img_count_gt.view(batch, h, w).contiguous()
        right_img_count_gt = right_img_count_gt.view(batch, h, w).contiguous()
        left_img_gt = left_img_gt.view(batch, h, w, channel).contiguous()
        right_img_gt = right_img_gt.view(batch, h, w, channel).contiguous()

        time_sum = 0
        for loop_idx in range(loops):
            left_img_count = torch.zeros(batch, h, w, dtype=torch.int32, device=device)
            right_img_count = torch.zeros(batch, h, w, dtype=torch.int32, device=device)
            left_img_stack = image.new_zeros(size=(batch, h, w, channel))
            right_img_stack = image.new_zeros(size=(batch, h, w, channel))

            # Call CUDA function
            torch.cuda.synchronize()
            time_start = time.time()

            DualPixel.DepthMerge(
                image,
                depth,
                left_img_count,
                right_img_count,
                left_img_stack,
                right_img_stack
            )

            torch.cuda.synchronize()
            duration = time.time() - time_start
            time_sum += duration if (loop_idx > 0) else time_sum

        print(f'Average CUDA time: {time_sum * 1.0e3 / (loops - 1):.4f}ms.')

        # Check correctness
        print(
            f"Error, left count: {(left_img_count - left_img_count_gt).abs().max().cpu().numpy():d}" \
            f", right count: {(right_img_count - right_img_count_gt).abs().max().cpu().numpy():d}" \
            f"; left img: {(left_img_stack - left_img_gt).abs().max().cpu().numpy():.4f}" \
            f", right img: {(left_img_stack - left_img_gt).abs().max().cpu().numpy():.4f}."
        )