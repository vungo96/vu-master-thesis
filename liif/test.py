import argparse
import os
import math
from functools import partial

from torchvision import transforms
import torch.nn as nn

import math
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

def save_images_to_dir(out_dir, inp, pred, gt, step=0, tag=""):
    transforms.ToPILImage()(inp[0]).save(f'{out_dir}/{step}_inp.png')
    transforms.ToPILImage()(pred[0]).save(f'{out_dir}/{step}_pred.png')
    transforms.ToPILImage()(gt[0]).save(f'{out_dir}/{step}_gt.png')

def batched_predict(model, inp, coord, cell, bsize, inp_scale=None):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            if inp_scale is not None:
                pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :], inp_scale[:, ql: qr])
            else:
                pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, max_scale=4,
              verbose=False, device="cuda", writer=None, epoch=0, out_dir=None, window_size=0):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device)

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

    val_res_psnr = utils.Averager()
    val_res_ssim = utils.Averager()
    val_res_lpips = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for i, batch in enumerate(pbar):
        for k, v in batch.items():
            batch[k] = v.to(device)

        inp = (batch['inp'] - inp_sub) / inp_div

        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]
            
            coord = utils.make_coord((scale*(h_old+h_pad), scale*(w_old+w_pad))).unsqueeze(0).cuda()
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0
            
            coord = batch['coord']
            cell = batch['cell']

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(model, inp,
                                coord, cell*max(scale/max_scale, 1), eval_bsize)
            
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None:  # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            
            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]
        else:
            # prep data
            sample_patch_size = round(math.sqrt(pred.shape[-2]))
            pred = pred.reshape(pred.shape[0], sample_patch_size, sample_patch_size, 3).permute(0, 3, 1, 2)
            batch['gt'] = batch['gt'].reshape(pred.shape[0], sample_patch_size, sample_patch_size, 3).permute(0, 3, 1, 2)
            
        # TODO: remove the hardcoded 5
        if out_dir is not None and (i % 5) == 0:
            save_images_to_dir(out_dir, batch['inp'], pred, batch['gt'], step=i)

        # TODO: test psnr with coordinates
        res_psnr = metric_psnr(pred, batch['gt']) 
        res_ssim = metric_ssim(pred, batch['gt'])
        res_lpips = metric_lpips(pred, batch['gt'])
        val_res_psnr.add(res_psnr.item(), inp.shape[0])
        val_res_ssim.add(res_ssim.item(), inp.shape[0])
        val_res_lpips.add(res_lpips.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val psnr {:.4f} | ssim loss {:.4f} | lpips loss {:.4f}'.format(val_res_psnr.item(), val_res_ssim.item(), val_res_lpips.item()))

    return val_res_psnr.item(), val_res_ssim.item(), val_res_lpips.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--window', default=0)
    args = parser.parse_args()

    print("Tag: ", args.tag)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Run on device: ", device)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # writer for saving qualitative results
    save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save_test', save_name)
    log, writer = utils.set_save_path(save_path)

    if args.out_dir is not None:
        out_dir = os.path.join(args.out_dir, save_name)
        os.makedirs(out_dir, exist_ok=True)
        

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8, pin_memory=True)

    model_spec = torch.load(
        args.model, map_location="cpu")['model']
    model = models.make(model_spec, load_sd=True).to(device)

    if n_gpus > 1:
        print("Use multiple gpus.")
        model = nn.parallel.DataParallel(model)

    res_psnr, res_ssim, res_lpips = eval_psnr(loader, model,
                    data_norm=config.get('data_norm'),
                    eval_type=config.get('eval_type'),
                    eval_bsize=config.get('eval_bsize'),
                    verbose=True,
                    device=device,
                    writer=writer,
                    out_dir=args.out_dir,
                    window_size=int(args.window))
    print('result psnr: {:.4f}'.format(res_psnr))
    print('result ssim: {:.4f}'.format(res_ssim))
    print('result lpips: {:.4f}'.format(res_lpips))
