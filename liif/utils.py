import os
import time
import shutil
import random

import torch
import cv2
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms

def save_image_to_dir(img, out_dir='test', step=0):
    transforms.ToPILImage()(img).save(f'{out_dir}/{step}.png')

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

def calc_ssim(sr, hr, dataset=None, scale=1):
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        sr = sr[..., shave:-shave, shave:-shave]
        hr = hr[..., shave:-shave, shave:-shave]

        return structural_similarity_index_measure(sr, hr)
    else:
        return structural_similarity_index_measure(sr, hr)

def calc_lpips(sr, hr, dataset=None, scale=1, device='cuda'):
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            lpips_net = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
            lpips = lpips_net(sr*2-1, hr*2-1)

            return lpips
        sr = sr[..., shave:-shave, shave:-shave]
        hr = hr[..., shave:-shave, shave:-shave]

        lpips_net = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        with torch.no_grad():
            lpips = lpips_net(sr*2-1, hr*2-1)

        return lpips
    else:
        lpips_net = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        with torch.no_grad():
            lpips = lpips_net(sr*2-1, hr*2-1)

        return lpips

def choose_scale_exponentially(min_scale, max_scale):
    scales = list(range(min_scale, max_scale + 1))
    
    # Calculate the weights exponentially
    weights = [2**(scale - min_scale) for scale in scales]
    
    # Use random.choices to select a scale based on the defined weights
    chosen_scale = random.choices(scales, weights)[0]
    
    return chosen_scale

def Huber(input, target, delta=0.01, reduce=True):
    abs_error = torch.abs(input - target)
    quadratic = torch.clamp(abs_error, max=delta)

    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * torch.pow(quadratic, 2) + delta * linear
    
    if reduce:
        return torch.mean(losses)
    else:
        return losses

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
   
def get_random_coordinate_from_edges(image_tensor):
     # Convert the image tensor to a NumPy array
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Convert the grayscale image to 8-bit unsigned integer
    gray_image = np.uint8(gray_image * 255.0)

    # Apply Canny edge detection using OpenCV
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

    # Find the coordinates of the edge pixels
    edge_indices = np.argwhere(edges > 0)

    # Randomly select one of the edge coordinates
    random_index = np.random.randint(len(edge_indices))
    edge_coordinate = edge_indices[random_index]

    return edge_coordinate

def get_image_crop_start_coord(image, center_coord, p):
    # Calculate the height and width of the image
    center_row, center_col = center_coord[0], center_coord[1]
    height, width = image.size(-2), image.size(-1)

    # Calculate the starting and ending row indices of the crop
    start_row = center_row - p // 2
    end_row = start_row + p

    # Check if the crop exceeds the top or bottom boundaries of the image
    if start_row < 0:
        start_row = 0
        end_row = p
    elif end_row > height:
        end_row = height
        start_row = end_row - p

    # Calculate the starting and ending column indices of the crop
    start_col = center_col - p // 2
    end_col = start_col + p

    # Check if the crop exceeds the left or right boundaries of the image
    if start_col < 0:
        start_col = 0
        end_col = p
    elif end_col > width:
        end_col = width
        start_col = end_col - p

    # Perform the crop
    #cropped_image = image[:, start_row:end_row, start_col:end_col]

    return start_row, start_col



