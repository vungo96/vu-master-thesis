import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models

from models import register
from utils import make_coord
from .glean_styleganv2 import RRDBFeatureExtractor


@register('liif_glean_styleganv2')
class LiifGleanStyleGANv2(nn.Module):

    def __init__(self,
                 generator_spec,
                 in_size,
                 out_size,
                 img_channels=3,
                 rrdb_channels=64,
                 num_rrdbs=23,
                 style_channels=512,
                 edsr_channels=64, # remove later (not needed since we use RRDB encoder instead of EDSR)
                 imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, device='cpu'):
        super().__init__()
        self.device = device
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        # input size must be strictly smaller than output size
        if in_size >= out_size:
            raise ValueError('in_size must be smaller than out_size, but got '
                             f'{in_size} and {out_size}.')

        # latent bank (StyleGANv2), with weights being fixed
        self.generator = models.make(
            generator_spec, load_sd=True, prefix=True).to(self.device)
        self.generator.requires_grad_(False)

        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels

        # first RRDB encoder from GLEAN paper
        num_styles = int(np.log2(out_size)) * 2 - 2
        encoder_res = [2**i for i in range(int(np.log2(in_size)), 1, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    img_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        for res in encoder_res:
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder.append(block)

        # additional modules for StyleGANv2
        self.fusion_out = nn.ModuleList()
        #self.fusion_skip = nn.ModuleList()
        for res in encoder_res[::-1]:
            num_channels = channels[res]
            self.fusion_out.append(
                nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
            # self.fusion_skip.append(
            #    nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))

        # encoder for latent bank such that we can feed into MLP
        encoder_bank_res = [
            2**i for i in range(int(np.log2(out_size)), int(np.log2(in_size)), -1)]
        self.encoder_bank = nn.ModuleList()
        """ self.encoder_bank.append(
            nn.Sequential(
                # TODO: might remove this feature extractor
                RRDBFeatureExtractor(
                    channels[out_size], rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True))) """
        for res in encoder_bank_res:
            # TODO: addapt to channels of edsr in liif?
            in_channels = channels[res]
            out_channels = channels[res // 2]
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.encoder_bank.append(block)

        # additional modules for fusing first encoder and latent bank to second encoder
        self.fusion_bank_out = nn.ModuleList()
        #self.fusion_bank_skip = nn.ModuleList()
        for res in encoder_bank_res:
            num_channels = channels[res]
            self.fusion_bank_out.append(
                nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
            # self.fusion_bank_skip.append(
            #    nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))

        # MLP takes features from last layer of 2nd encoder
        if imnet_spec is not None:
            # TODO: concat features of encoder and generator and different layers
            imnet_in_dim = out_channels
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2  # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, lq):
        h, w = lq.shape[2:]
        if h != self.in_size or w != self.in_size:
            raise AssertionError(
                f'Spatial resolution must equal in_size ({self.in_size}).'
                f' Got ({h}, {w}).')

        # RRDB encoder
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]

        latent = encoder_features[0].view(lq.size(0), -1, self.style_channels)
        encoder_features = encoder_features[1:]

        # generator (latent bank)
        injected_noise = [
            getattr(self.generator, f'injected_noise_{i}')
            for i in range(self.generator.num_injected_noises)
        ]
        # 4x4 stage
        out = self.generator.constant_input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0])
        # skip = self.generator.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher res
        generator_features = []
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2],
                injected_noise[1::2], injected_noise[2::2],
                self.generator.to_rgbs):

            # feature fusion by channel-wise concatenation
            if out.size(2) <= self.in_size:
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]

                out = torch.cat([out, feat], dim=1)
                out = self.fusion_out[fusion_index](out)

                #skip = torch.cat([skip, feat], dim=1)
                #skip = self.fusion_skip[fusion_index](skip)

            # original StyleGAN operations
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            #skip = to_rgb(out, latent[:, _index + 2], skip)

            # store features for decoder
            if out.size(2) > self.in_size:
                generator_features.append(out)

            _index += 2

        # Concat features of second encoder and latent bank for same resolution layers
        # EDSR from liif paper is replaced by RRDB encoder from GLEAN paper
        out = generator_features[-1]
        encoder_bank_features = []
        for i, block in enumerate(self.encoder_bank):
            if i > 0 and i < len(generator_features):
                out = torch.cat(
                    [out, generator_features[len(generator_features)-i-1]], dim=1)
                out = self.fusion_bank_out[i](out)
            out = block(out)
            encoder_bank_features.append(out)
        self.feat = encoder_bank_features[-1]

        return self.feat

    def query_rgb(self, coord, cell=None):
        # copied from original repo take care of querying rgb for coordinates
        # see paper for more information on feature unfolding, local ensemble and cell decoding
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        # feature unfolding
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        # local ensemble
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(self.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                # cell decoding for taking shape of the query pixel as addtional input
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        # local ensemble
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
