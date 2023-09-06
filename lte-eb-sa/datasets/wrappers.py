import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples, get_random_coordinate_from_edges, get_image_crop_start_coord


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2]  # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # sample q coordinates
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, scale_adaptive=None, plot_scales=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.scale_adaptive = scale_adaptive
        self.plot_scales = plot_scales
        if scale_max is None and scale_adaptive is None:
            self.scale_max = scale_min

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        min_dim = min(img.size(1), img.size(2))
        # scale-adaptive
        if self.scale_adaptive is not None:
            scale_max = min_dim // self.inp_size
            if self.scale_max is not None:
                if self.scale_max < scale_max:
                    scale_max = self.scale_max
        s = random.uniform(self.scale_min, scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)
                      ]  # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            if self.crop_from_edges is not None:
                coord_from_edge = get_random_coordinate_from_edges(img)
                x0, y0 = get_image_crop_start_coord(img, coord_from_edge, w_hr)
            else:
                x0 = random.randint(0, img.shape[-2] - w_hr)
                y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # sample q coordinates
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        rtn_dict = {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
        }

        if self.plot_scales:
            rtn_dict['scale'] = torch.tensor(s)
            rtn_dict['scale_max'] = torch.tensor(self.scale_max)

        return rtn_dict


@register('sr-implicit-downsampled-collate-batch')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_sizes=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, plot_scales=False, crop_from_edges=None):
        self.dataset = dataset
        self.inp_sizes = inp_sizes
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.plot_scales = plot_scales
        self.crop_from_edges = crop_from_edges

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        return img

    def collate_batch(self, batch):

        rtn_lists = {
            'inp': [],
            'coord': [],
            'cell': [],
            'gt': [],
            'inp_scale': [],
            'scale': [],
            'scale_max': []
        }

        inp_idx = random.randint(0, len(self.inp_sizes)-1)
        inp_size = self.inp_sizes[inp_idx]

        for img in batch:
            min_dim = min(img.size(1), img.size(2))
            # scale-adaptive
            scale_max = min_dim // inp_size

            if self.scale_max is not None:
                if self.scale_max < scale_max:
                    scale_max = self.scale_max
            s = random.uniform(self.scale_min, scale_max)

            w_lr = inp_size
            w_hr = round(w_lr * s)
            if self.crop_from_edges is not None:
                coord_from_edge = get_random_coordinate_from_edges(img)
                x0, y0 = get_image_crop_start_coord(img, coord_from_edge, w_hr)
            else:
                x0 = random.randint(0, img.shape[-2] - w_hr)
                y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

            if self.augment:
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                dflip = random.random() < 0.5

                def augment(x):
                    if hflip:
                        x = x.flip(-2)
                    if vflip:
                        x = x.flip(-1)
                    if dflip:
                        x = x.transpose(-2, -1)
                    return x

                crop_lr = augment(crop_lr)
                crop_hr = augment(crop_hr)

            hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

            # sample q coordinates
            if self.sample_q is not None:
                sample_lst = np.random.choice(
                    len(hr_coord), self.sample_q, replace=False)
                hr_coord = hr_coord[sample_lst]
                hr_rgb = hr_rgb[sample_lst]

            cell = torch.ones_like(hr_coord)
            cell[:, 0] *= 2 / crop_hr.shape[-2]
            cell[:, 1] *= 2 / crop_hr.shape[-1]

            inp_scale = torch.ones(hr_coord.shape[-2])
            inp_scale[:] *= s

            rtn_lists['inp'].append(crop_lr)
            rtn_lists['coord'].append(hr_coord)
            rtn_lists['cell'].append(cell)
            rtn_lists['gt'].append(hr_rgb)
            rtn_lists['inp_scale'].append(inp_scale)

            if self.plot_scales:
                rtn_lists['scale'].append(s)
                rtn_lists['scale_max'].append(scale_max)

        rtn_dict = {
            'inp': torch.stack(rtn_lists['inp'], dim=0),
            'coord': torch.stack(rtn_lists['coord'], dim=0),
            'cell': torch.stack(rtn_lists['cell'], dim=0),
            'gt': torch.stack(rtn_lists['gt'], dim=0),
            'inp_scale': torch.stack(rtn_lists['inp_scale'], dim=0)
        }

        if self.plot_scales:
            rtn_dict['scale'] = torch.tensor(rtn_lists['scale'])
            rtn_dict['scale_max'] = torch.tensor(rtn_lists['scale_max'])

        return rtn_dict
