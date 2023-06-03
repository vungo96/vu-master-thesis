import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

import numpy as np

@register('lte-bn')
class LTE(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256, scale_aware_phase=None, scale_aware_mlp=None, device='cuda'):
        super().__init__()        
        self.device = device
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.scale = nn.Sequential(
            nn.Linear(1, hidden_dim//2, bias=False),
            nn.ReLU(),
        )
        self.scale_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU()
        )
        self.phase = nn.Linear(2, hidden_dim//2, bias=False)
        self.scale_aware_phase = scale_aware_phase
        self.scale_aware_mlp = scale_aware_mlp

        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})
        self.bn1 = nn.BatchNorm1d(3)

    def gen_feat(self, inp):
        self.inp = inp
        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).to(self.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])
        
        self.feat = self.encoder(inp)
        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell=None, scale=None):
        feat = self.feat
        coef = self.coeff
        freq = self.freqq

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6 

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                
                # basis generation
                bs, q = coord.shape[:2]
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                if scale is not None and self.scale_aware_phase is not None:
                    q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1) + self.scale(scale.view(bs * q, -1)).view(bs, q, -1)
                else:
                    q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)

                inp = torch.mul(q_coef, q_freq)       

                if self.scale_aware_mlp is not None and scale is not None:
                    inp += self.scale_mlp(scale.view(bs * q, -1)).view(bs, q, -1)  

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return ret

    def forward(self, inp, coord, cell, scale=None):
        self.gen_feat(inp)
        rgbs = self.query_rgb(coord, cell, scale)

        # Reshape the batch to (batch_size, c, num_coordinates)
        reshaped_rgbs = rgbs.permute(0, 2, 1)

        # Perform batch normalization
        normalized_rgbs = self.bn1(reshaped_rgbs)

        # Reshape the normalized batch back to the original shape
        return normalized_rgbs.permute(0, 2, 1)