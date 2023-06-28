""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

import argparse
import os

import pickle
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models import unetd
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import datasets
import models
import utils
from test import eval_psnr, save_images_to_dir
import LPIPS.models.dist_model as dm


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    if spec['collate_batch']:
        # TODO: print stuff here as well
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                            shuffle=(tag == 'train'), num_workers=12, collate_fn=dataset.collate_batch, pin_memory=True)
    else:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                            shuffle=(tag == 'train'), num_workers=12, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def prepare_training():
    gan_based = config.get('gan_based')
    if gan_based is None:
        model_D = None
        optimizer_D = None
        params_D = None
        lr_scheduler_D = None

    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(
            sv_file['model'], load_sd=True).to(device)
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = utils.make_optimizer(
            params, sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
            
        if gan_based is not None:
            sv_file_D = torch.load(config['resume_D'])
            model_D = models.make(
            sv_file_D['model'], load_sd=True).to(device)
            log('model_D: #params={}'.format(utils.compute_num_params(model_D, text=True)))

            params_D = list(filter(lambda p: p.requires_grad, model_D.parameters()))
            optimizer_D = utils.make_optimizer(
            params_D, sv_file_D['optimizer'], load_sd=True)
            lr_scheduler_D = MultiStepLR(optimizer_D, **config['multi_step_lr_D'])
            for _ in range(epoch_start - 1):
                lr_scheduler_D.step()
    elif config.get('pretrained') is not None:
        print("Use pretrained model.")
        if gan_based is not None:
            model_D = models.make(config['model_D']).to(device)
            log('model_D: #params={}'.format(utils.compute_num_params(model_D, text=True)))

            params_D = list(filter(lambda p: p.requires_grad, model_D.parameters()))
            optimizer_D = utils.make_optimizer(params_D, config['optimizer_D'])
            lr_scheduler_D = MultiStepLR(optimizer_D, **config['multi_step_lr_D'])
        sv_file = torch.load(config['pretrained'])
        model = models.make(
            sv_file['model'], load_sd=True).to(device)
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = utils.make_optimizer(
            params, config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
    else:
        if gan_based is not None:
            model_D = models.make(config['model_D']).to(device)
            log('model_D: #params={}'.format(utils.compute_num_params(model_D, text=True)))

            params_D = list(filter(lambda p: p.requires_grad, model_D.parameters()))
            optimizer_D = utils.make_optimizer(params_D, config['optimizer_D'])
            lr_scheduler_D = MultiStepLR(optimizer_D, **config['multi_step_lr_D'])

        model = models.make(config['model']).to(device)
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = utils.make_optimizer(
            params, config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return epoch_start, model, optimizer, params, lr_scheduler, model_D, optimizer_D, params_D, lr_scheduler_D

def add_scales_to_dict(scales, scales_max):
    for scale in scales.tolist():
        if round(scale) in scale_freq.keys():
            scale_freq[round(scale)] += 1
        else:
            scale_freq[round(scale)] = 1
    
    for scale_max in scales_max.tolist():
        if scale_max in scale_max_freq.keys():
            scale_max_freq[scale_max] += 1
        else:
            scale_max_freq[scale_max] = 1

def train(train_loader, model, optimizer, params, gradient_accumulation_steps, model_D=None, optimizer_D=None, params_D=None, l_adv=0.001, l_fm=1, l_lpips=0.000001, n_mix = 0, loss_fn='l1', lpips_net=None):
    model.train()
    loss_fn = nn.L1Loss()
    if loss_fn == 'huber':
        print("Huber as loss fn")
        loss_fn = utils.Huber
        
    train_loss = utils.Averager()
    train_loss_D = utils.Averager()

    if model_D is not None:
        model_D.train()

    # normalize data
    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device)

    # train batches
    for step, batch in tqdm(enumerate(train_loader), leave=False, desc='train', total=len(train_loader)):
        for k, v in batch.items():
            batch[k] = v.to(device)

        inp = (batch['inp'] - inp_sub) / inp_div
        gt = (batch['gt'] - gt_sub) / gt_div 

        if model_D is not None:
            optimizer.zero_grad()
            optimizer_D.zero_grad()

            pred = model(inp, batch['coord'], batch['cell']).detach()
        
            # TODO: check if this works
            pred.clamp_(-1, 1)

            # prep data
            sample_patch_size = round(math.sqrt(pred.shape[-2]))
            pred_img = pred.reshape(pred.shape[0], sample_patch_size, sample_patch_size, 3).permute(0, 3, 1, 2)
            gt_img = gt.reshape(pred.shape[0], sample_patch_size, sample_patch_size, 3).permute(0, 3, 1, 2)

            # TODO: remove later only for debugging
            #pred_img = pred_img * gt_div + gt_sub
            #pred_img.clamp_(0, 1)
            #gt_img = gt_img * gt_div + gt_sub
            #gt_img.clamp(0,1)
            #save_images_to_dir("test_images", inp, pred_img, gt_img, step=step)

            e_S, d_S, _, _ = model_D(pred_img)
            e_H, d_H, _, _ = model_D(gt_img)

            # D Loss, for encoder end and decoder end
            loss_D_Enc_S = torch.nn.ReLU()(1.0 + e_S).mean()
            loss_D_Enc_H = torch.nn.ReLU()(1.0 - e_H).mean()

            loss_D_Dec_S = torch.nn.ReLU()(1.0 + d_S).mean()
            loss_D_Dec_H = torch.nn.ReLU()(1.0 - d_H).mean()

            loss_D = loss_D_Enc_H + loss_D_Dec_H

            # CutMix for consistency loss
            batch_S_CutMix = pred_img.clone()

            # probability of doing cutmix
            # p_mix = i / 100000
            #if p_mix > 0.5:
            # TODO: change?
            p_mix = 0.5

            if torch.rand(1) <= p_mix:
                n_mix += 1
                r_mix = torch.rand(1)   # real/fake ratio

                bbx1, bby1, bbx2, bby2 = utils.rand_bbox(batch_S_CutMix.size(), r_mix)
                batch_S_CutMix[:, :, bbx1:bbx2, bby1:bby2] = gt_img[:, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                r_mix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_S_CutMix.size()[-1] * batch_S_CutMix.size()[-2]))

                e_mix, d_mix, _, _ = model_D( batch_S_CutMix )

                loss_D_Enc_S = torch.nn.ReLU()(1.0 + e_mix).mean()
                loss_D_Dec_S = torch.nn.ReLU()(1.0 + d_mix).mean()

                d_S[:,:,bbx1:bbx2, bby1:bby2] = d_H[:,:,bbx1:bbx2, bby1:bby2]
                loss_D_Cons = F.mse_loss(d_mix, d_S)

                loss_D += loss_D_Cons
                # l_accum[5] += torch.mean(loss_D_Cons).item()

            loss_D += loss_D_Enc_S + loss_D_Dec_S

            # Update
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(params_D, 0.1)
            optimizer_D.step()
            optimizer_D.zero_grad()

            train_loss_D.add(loss_D.item())

            # for monitoring
            #l_accum[0] += loss_D.item()
            #l_accum[1] += torch.mean(e_H).item()
            #l_accum[2] += torch.mean(e_S).item()
            #l_accum[3] += torch.mean(d_H).item()
            #l_accum[4] += torch.mean(d_S).item()

        optimizer.zero_grad()

        pred = model(inp, batch['coord'], batch['cell'])
    
        # TODO: check if this works
        pred.clamp_(-1, 1)
            
        loss = loss_fn(pred, gt)
        # loss_Pixel = utils.Huber(pred, gt)
        # loss_G = loss_Pixel

        if model_D is not None:
            # prep data
            sample_patch_size = round(math.sqrt(pred.shape[-2]))
            pred_img = pred.reshape(pred.shape[0], sample_patch_size, sample_patch_size, 3).permute(0, 3, 1, 2)
            gt_img = gt.reshape(pred.shape[0], sample_patch_size, sample_patch_size, 3).permute(0, 3, 1, 2)

            # LPIPS loss
            #loss_LPIPS, _ = model_LPIPS.forward_pair(batch_H*2-1, batch_S*2-1)
            #loss_LPIPS = torch.mean(loss_LPIPS) * L_LPIPS
            lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
            loss_LPIPS = lpips(pred_img, gt_img) * l_lpips

            # FM and GAN losses
            e_S, d_S, e_Ss, d_Ss = model_D(pred_img)
            _, _, e_Hs, d_Hs = model_D(gt_img)

            # FM loss
            loss_FMs = []
            for f in range(3):
                loss_FMs += [utils.Huber(e_Ss[f], e_Hs[f])]
                loss_FMs += [utils.Huber(d_Ss[f], d_Hs[f])]
            loss_FM = torch.mean(torch.stack(loss_FMs)) * l_fm

            # GAN loss
            loss_Advs = []
            loss_Advs += [torch.nn.ReLU()(1.0 - e_S).mean() * l_adv]
            loss_Advs += [torch.nn.ReLU()(1.0 - d_S).mean() * l_adv]
            loss_Adv = torch.mean(torch.stack(loss_Advs))

            loss += loss_LPIPS + loss_FM + loss_Adv

            # For monitoring
            #l_accum[7] += loss_LPIPS.item()
            #l_accum[8] += loss_FM.item()
            #l_accum[9] += loss_Adv.item()

        loss = loss / gradient_accumulation_steps
        loss.backward()
        # TODO: change back
        torch.nn.utils.clip_grad_norm_(params, 0.1)

        if step % gradient_accumulation_steps == 0:
            train_loss.add(loss.item())
            optimizer.step()

            pred = None; loss = None

        if 'scale' in batch.keys() and 'scale_max' in batch.keys():
            add_scales_to_dict(batch['scale'], batch['scale_max'])
            
    return train_loss.item()


def main(config_, save_path):
    global config, log, writer, device, scale_freq, scale_max_freq
    scale_freq = {}
    scale_max_freq = {}
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # get data loaders from config
    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    print("Config: ", config)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    # TODO: change this back
    num_iter_per_epoch = math.ceil(len(train_loader.dataset) / config['train_dataset']['batch_size']) # config['train_dataset']['batch_size'])
    iter_max = config['iter_max']
    epoch_max = math.ceil(iter_max / num_iter_per_epoch)
    epoch_val = math.floor(config.get('iter_val') / num_iter_per_epoch)
    epoch_save = math.floor(config.get('iter_save') / num_iter_per_epoch)
    config['multi_step_lr']['milestones'] = [math.floor(milestone / num_iter_per_epoch) for milestone in config['multi_step_lr']['milestones']]
    print('len dataset: ', len(train_loader.dataset))
    print('epoch_max:', epoch_max)
    print('epoch_val', epoch_val)
    print('epoch_save', epoch_save)
    print('milestones', config['multi_step_lr'])

    # Enable running on cpu as well
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Run on device: ", device)

    gan_based = config.get('gan_based')
    print('Gan-based:', gan_based)

    ## LPIPS
    lpips_net = dm.DistModel()
    lpips_net.initialize(model='net-lin',net='alex',use_gpu=True)

    # get model, optimizer, and lr_scheduler from config
    epoch_start, model, optimizer, params, lr_scheduler, model_D, optimizer_D, params_D, lr_scheduler_D = prepare_training()

    if n_gpus > 1:
        print("Use multiple gpus.")
        model = nn.parallel.DataParallel(model)
        if gan_based is not None:
            print("Use multiple gpus for discriminator.")
            model_D = nn.parallel.DataParallel(model_D)
            lpips_net = nn.parallel.DataParallel(lpips_net)

    max_val_v = -1e18

    gradient_accumulation_steps = config.get('gradient_accumulation_steps')

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # train epoch
        if gan_based is not None:
            train_loss = train(train_loader, model, optimizer, params, gradient_accumulation_steps, model_D, optimizer_D, params_D, loss_fn=config.get('loss_fn'), lpips_net=lpips_net)
        else:
            # TODO: make sure that optimizer params are correct for clip
            train_loss = train(train_loader, model, optimizer, params, gradient_accumulation_steps, loss_fn=config.get('loss_fn'))
        if lr_scheduler is not None:
            lr_scheduler.step()
        if gan_based is not None and lr_scheduler_D is not None:
            lr_scheduler_D.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
            if gan_based is not None:
                model_D_ = model_D.module
        else:
            model_ = model
            if gan_based is not None:
                model_D_ = model_D
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }
        # TODO: chang eback --> should just be faster
        # torch.save(sv_file, os.path.join(save_path, 'iteration-last.pth'))

        if gan_based is not None:
            model_spec_D = config['model_D']
            model_spec_D['sd'] = model_D_.state_dict()
            optimizer_spec_D = config['optimizer_D']
            optimizer_spec_D['sd'] = optimizer_D.state_dict()
            sv_file_D = {
                'model': model_spec_D,
                'optimizer': optimizer_spec_D,
                'epoch': epoch
            }
            torch.save(sv_file_D, os.path.join(save_path, 'd_iteration-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                       os.path.join(save_path, 'iteration-{}.pth'.format(epoch*num_iter_per_epoch)))
            if gan_based is not None:
                torch.save(sv_file_D, 
                           os.path.join(save_path, 'd_iteration-{}.pth'.format(epoch*num_iter_per_epoch)))
            # save scale_freq dict
            if scale_freq:
                with open(os.path.join(save_path, 'scale_freq.pickle'), "wb") as f:
                    pickle.dump(scale_freq, f)
            if scale_max_freq:
                with open(os.path.join(save_path, 'scale_max_freq.pickle'), "wb") as f:
                    pickle.dump(scale_max_freq, f)

        # validate
        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model

            val_res_psnr, val_res_ssim, val_res_lpips = eval_psnr(val_loader, model_,
                                data_norm=config['data_norm'],
                                eval_type=config.get('eval_type'),
                                eval_bsize=config.get('eval_bsize'),
                                device=device, 
                                writer=writer,
                                epoch=epoch)

            log_info.append('val psnr {:.4f} | ssim loss {:.4f} | lpips loss {:.4f}'.format(val_res_psnr, val_res_ssim, val_res_lpips))
            writer.add_scalars('psnr', {'val': val_res_psnr}, epoch)
            writer.add_scalars('ssim', {'val': val_res_ssim}, epoch)
            writer.add_scalars('lpips', {'val': val_res_lpips}, epoch)

            if val_res_psnr > max_val_v:
                max_val_v = val_res_psnr
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    print('Tag: ', args.tag)

    main(config, save_path)
