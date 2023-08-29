import math
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from PIL import ImageFile, Image
from functools import partial

import datasets
import models
import utils
import os

device = 'cuda'
eval_type = 'div2k-4'
pred_folder = 'test_images/compute_patch_metrics/pred'
gt_folder = 'test_images/compute_patch_metrics/gt'

transform = transforms.Compose([
        transforms.ToTensor(),
    ])

pred_files = sorted([file for file in os.listdir(pred_folder) if file.endswith('.png')])  # Change the extension to match your image format
pred_images = [transform(Image.open(os.path.join(pred_folder, file)).convert('RGB')) for file in pred_files]

gt_files = sorted([file for file in os.listdir(gt_folder) if file.endswith('.png')])  # Change the extension to match your image format
gt_images = [transform(Image.open(os.path.join(gt_folder, file)).convert('RGB')) for file in gt_files]

print("eval_type:", eval_type)
if eval_type is None:
    metric_psnr = utils.calc_psnr
    metric_ssim = utils.calc_ssim
    metric_lpips = utils.calc_lpips
elif eval_type.startswith('div2k'):
    scale = int(eval_type.split('-')[1])
    metric_psnr = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    metric_ssim = partial(utils.calc_ssim, dataset='div2k', scale=scale)
    metric_lpips = partial(utils.calc_lpips, dataset='div2k', scale=scale, device=device)
elif eval_type.startswith('benchmark'):
    scale = int(eval_type.split('-')[1])
    metric_psnr = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    metric_ssim = partial(utils.calc_ssim, dataset='benchmark', scale=scale)
    metric_lpips = partial(utils.calc_lpips, dataset='benchmark', scale=scale, device=device)
else:
    raise NotImplementedError

res_psnr = 0
res_lpips = 0

for i in range(len(pred_images)):
    sr = pred_images[i]
    # TODO: change?
    hr = gt_images[0]

    if sr.shape != hr.shape:
        print('shave to same shape')
        print(sr.shape)
        print(hr.shape)
        # Find the dimensions of the smaller image
        H_small = min(sr.shape[1], hr.shape[1])
        W_small = min(sr.shape[2], hr.shape[2])

        # Calculate the number of pixels to shave off from each side
        shave_H = (hr.shape[1] - H_small) // 2
        shave_W = (hr.shape[2] - W_small) // 2

        # Shave the images to match the dimensions of the smaller image
        # sr = sr[:, shave_H:-shave_H, shave_W:-shave_W]
        if shave_H != 0:
            hr = hr[..., shave_H:-shave_H, :]
        if shave_W != 0:
            hr = hr[..., :, shave_W:-shave_W]
        #hr = hr[..., shave_H:-shave_H, shave_W:-shave_W]

    sr_batch = sr.unsqueeze(0).to('cuda')
    hr_batch = hr.unsqueeze(0).to('cuda')
    print(metric_psnr(sr_batch, hr_batch))
    print(metric_lpips(sr_batch, hr_batch))
    res_psnr += metric_psnr(sr_batch, hr_batch)
    res_lpips += metric_lpips(sr_batch, hr_batch)

print('psnr:', res_psnr / len(pred_images))
print('lpips:', res_lpips / len(pred_images))