# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner.checkpoint import _load_checkpoint_with_prefix

import copy


models = {}


def remove_unexpected_keys(state_dict, model_dict):
    state_dict_keys = set(state_dict.keys())
    for key in state_dict_keys:
        if key not in model_dict.keys():
            state_dict.pop(key)
    return state_dict


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False, prefix=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    if load_sd:
        state_dict = torch.load(model_spec['sd'], map_location='cpu')
        if prefix:
            state_dict = state_dict[model_spec['prefix']]
        # state_dict = remove_unexpected_keys(state_dict, model.state_dict())
        model.load_state_dict(state_dict)
    return model


# Copyright (c) OpenMMLab. All rights reserved.
def pixel_unshuffle(x, scale):
    """Down-sample by pixel unshuffle.

    Args:
        x (Tensor): Input tensor.
        scale (int): Scale factor.

    Returns:
        Tensor: Output tensor.
    """

    b, c, h, w = x.shape
    if h % scale != 0 or w % scale != 0:
        raise AssertionError(
            f'Invalid scale ({scale}) of pixel unshuffle for tensor '
            f'with shape: {x.shape}')
    h = int(h / scale)
    w = int(w / scale)
    x = x.view(b, c, h, scale, w, scale)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, -1, h, w)


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack."""
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x
