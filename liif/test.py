import argparse
import os
import math
from functools import partial

from torchvision import transforms
import torch.nn as nn

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

def save_images_to_dir(out_dir, inp, pred, gt, step=0):
    # bring pred and gt back to h x w
    h, w = inp.shape[2:]
    s = int(math.sqrt(pred.shape[1] // (h*w)))
    pred = pred.view(pred.shape[0], h*s, w*s, 3).permute(0, 3, 1, 2)
    gt = gt.view(gt.shape[0], h*s, w*s, 3).permute(0, 3, 1, 2)

    transforms.ToPILImage()(inp[0]).save(f'{out_dir}/{step}_inp.png')
    transforms.ToPILImage()(pred[0]).save(f'{out_dir}/{step}_pred.png')
    transforms.ToPILImage()(gt[0]).save(f'{out_dir}/{step}_gt.png')

def add_images_to_writer(writer, inp, pred, gt, step=0, tag=None):
    # bring pred and gt back to h x w
    h, w = inp.shape[2:]
    s = int(math.sqrt(pred.shape[1] // (h*w)))
    pred = pred.view(pred.shape[0], h*s, w*s, 3).permute(0, 3, 1, 2)
    gt = gt.view(gt.shape[0], h*s, w*s, 3).permute(0, 3, 1, 2)

    writer.add_images(f'Epoch {step} batch {tag} GT', inp, step)
    writer.add_images(f'Epoch {step} batch {tag} pred', pred, step)
    writer.add_images(f'Epoch {step} batch {tag} input', gt, step)
    writer.flush()


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
              verbose=False, device="cuda", writer=None, epoch=0, out_dir=None, scale_aware=None):
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
    if scale_aware is not None:
        inp_scale_max = data_norm['inp_scale_max']
    else:
        inp_scale_max = None

    print("eval_type:", eval_type)
    if eval_type is None:
        metric_psnr = utils.calc_psnr
        #metric_ssim = utils.calc_ssim
        #metric_lpips = utils.calc_lpips
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
    # TODO: change this to activate adding images
    first = False
    for i, batch in enumerate(pbar):
        for k, v in batch.items():
            batch[k] = v.to(device)

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, batch['coord'], batch['cell'])
        else:
            if inp_scale_max is not None:
                # TODO: normalize again
                inp_scale = (batch['inp_scale'] - 1) / (inp_scale_max - 1)
                pred = batched_predict(model, inp,
                                batch['coord'], batch['cell']*max(scale/max_scale, 1), eval_bsize, inp_scale)
            else:
                  
                pred = batched_predict(model, inp,
                                batch['coord'], batch['cell']*max(scale/max_scale, 1), eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        # save qualitative results
        if writer is not None and first:
            add_images_to_writer(writer, batch['inp'], pred, batch['gt'],
                                step=epoch, tag=str(i))
            first = False
        
        if out_dir is not None and (i % 5) == 0:
            save_images_to_dir(out_dir, batch['inp'], pred, batch['gt'], step=i)

        if eval_type is not None:  # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res_psnr = metric_psnr(pred, batch['gt'])
        val_res_psnr.add(res_psnr.item(), inp.shape[0])
        if eval_type is not None:
            res_ssim = metric_ssim(pred, batch['gt'])
            res_lpips = metric_lpips(pred, batch['gt'])
        
            val_res_ssim.add(res_ssim.item(), inp.shape[0])
            val_res_lpips.add(res_lpips.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val psnr {:.4f}'.format(val_res_psnr.item()))

    if eval_type is None:
        return val_res_psnr.item(), None, None

    return val_res_psnr.item(), val_res_ssim.item(), val_res_lpips.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
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

    # TODO: remove later
    if 'inp_scale_max' in config['data_norm'].keys():
        scale_aware = True
        print("Condition on scale")
    else:
        scale_aware = None

    res_psnr, res_ssim, res_lpips = eval_psnr(loader, model,
                    data_norm=config.get('data_norm'),
                    eval_type=config.get('eval_type'),
                    eval_bsize=config.get('eval_bsize'),
                    verbose=True,
                    device=device,
                    writer=writer,
                    out_dir=args.out_dir,
                    scale_aware=scale_aware)
    print('result psnr: {:.4f}'.format(res_psnr))
    print('result ssim: {:.4f}'.format(res_ssim))
    print('result lpips: {:.4f}'.format(res_lpips))
