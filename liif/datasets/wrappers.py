import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples, make_coord


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

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
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
                 augment=False, sample_q=None, plot_scales=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.plot_scales = plot_scales

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
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
            # sample_q can be set higher than inp_size^2
            sample_q = self.sample_q
            if self.sample_q > len(hr_coord):
                sample_q = len(hr_coord)
            sample_lst = np.random.choice(
                len(hr_coord), sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

            # fill up coordinates with same values
            ratio = self.sample_q / len(hr_coord)
            if ratio > 1:
                hr_coord = hr_coord.repeat((self.sample_q // len(hr_coord)) + 1, 1)[:self.sample_q]
                hr_rgb = hr_rgb.repeat((self.sample_q // len(hr_rgb)) + 1, 1)[:self.sample_q]

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


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        # sample q coordinates
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
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

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }  
    

@register('sr-implicit-downsampled-collate-batch')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_sizes=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, plot_scales=False, limit_scale=None):
        self.dataset = dataset
        self.inp_sizes = inp_sizes
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.plot_scales = plot_scales
        self.limit_scale = limit_scale

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
                'scale': [],
                'scale_max': []
            }

         # get minimum width or height of all images in one batch
        # min_dim = min([img.size(1) for img in batch] + [img.size(2) for img in batch])
        inp_idx = random.randint(0, len(self.inp_sizes)-1)
        inp_size = self.inp_sizes[inp_idx]
        # print('input_size: ', inp_size)

        for img in batch:
            min_dim = min(img.size(1), img.size(2))
            scale_max = min_dim // inp_size
            if self.limit_scale is not None:
                s = random.uniform(scale_max-self.limit_scale, scale_max)
            else:
                s = random.uniform(self.scale_min, scale_max)
            
            if self.scale_max is not None:
                s = random.uniform(self.scale_min, self.scale_max)

            w_lr = inp_size
            w_hr = round(w_lr * s)
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
                # sample_q can be set higher than inp_size^2
                sample_q = self.sample_q
                if self.sample_q > len(hr_coord):
                    sample_q = len(hr_coord)
                sample_lst = np.random.choice(
                    len(hr_coord), sample_q, replace=False)
                hr_coord = hr_coord[sample_lst]
                hr_rgb = hr_rgb[sample_lst]

                # fill up coordinates with same values
                ratio = self.sample_q / len(hr_coord)
                if ratio > 1:
                    hr_coord = hr_coord.repeat((self.sample_q // len(hr_coord)) + 1, 1)[:self.sample_q]
                    hr_rgb = hr_rgb.repeat((self.sample_q // len(hr_rgb)) + 1, 1)[:self.sample_q]

            cell = torch.ones_like(hr_coord)
            cell[:, 0] *= 2 / crop_hr.shape[-2]
            cell[:, 1] *= 2 / crop_hr.shape[-1]

            rtn_lists['inp'].append(crop_lr)
            rtn_lists['coord'].append(hr_coord)
            rtn_lists['cell'].append(cell)
            rtn_lists['gt'].append(hr_rgb)

            if self.plot_scales:
                rtn_lists['scale'].append(s)
                rtn_lists['scale_max'].append(scale_max)
        
        rtn_dict = {
            'inp': torch.stack(rtn_lists['inp'], dim=0),
            'coord': torch.stack(rtn_lists['coord'], dim=0),
            'cell': torch.stack(rtn_lists['cell'], dim=0),
            'gt': torch.stack(rtn_lists['gt'], dim=0),
        }
        
        if self.plot_scales:
            rtn_dict['scale'] = torch.tensor(rtn_lists['scale'])
            rtn_dict['scale_max'] = torch.tensor(rtn_lists['scale_max'])
                
        return rtn_dict
    

@register('sr-implicit-downsampled-fast-collate-batch')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        return img
    
    def collate_batch(self, batch):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        rtn_lists = { 
                'inp': [],
                'coord': [],
                'cell': [],
                'gt': [],
                'scale': [],
                'scale_max': []
            }

         # get minimum width or height of all images in one batch
        # min_dim = min([img.size(1) for img in batch] + [img.size(2) for img in batch])
        inp_idx = random.randint(0, len(self.inp_sizes)-1)
        inp_size = self.inp_sizes[inp_idx]
        # print('input_size: ', inp_size)

        for img in batch:
            min_dim = min(img.size(1), img.size(2))
            scale_max = min_dim // inp_size
            if self.limit_scale is not None:
                s = random.uniform(scale_max-self.limit_scale, scale_max)
            else:
                s = random.uniform(self.scale_min, scale_max)
            
            if self.scale_max is not None:
                s = random.uniform(self.scale_min, self.scale_max)

            h_lr = inp_size
            w_lr = inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
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

            hr_coord = make_coord([h_hr, w_hr], flatten=False)
            hr_rgb = crop_hr
            
            # if self.inp_size is not None:
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)
            
            cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

            rtn_lists['inp'].append(crop_lr)
            rtn_lists['coord'].append(hr_coord)
            rtn_lists['cell'].append(cell)
            rtn_lists['gt'].append(hr_rgb)

            if self.plot_scales:
                rtn_lists['scale'].append(s)
                rtn_lists['scale_max'].append(scale_max)
        
        rtn_dict = {
            'inp': torch.stack(rtn_lists['inp'], dim=0),
            'coord': torch.stack(rtn_lists['coord'], dim=0),
            'cell': torch.stack(rtn_lists['cell'], dim=0),
            'gt': torch.stack(rtn_lists['gt'], dim=0),
        }
        
        if self.plot_scales:
            rtn_dict['scale'] = torch.tensor(rtn_lists['scale'])
            rtn_dict['scale_max'] = torch.tensor(rtn_lists['scale_max'])
                
        return rtn_dict