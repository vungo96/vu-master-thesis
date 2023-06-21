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
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils
from test import eval_psnr


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
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(
            sv_file['model'], load_sd=True).to(device)
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    elif config.get('pretrained') is not None:
        print("Use pretrained model.")
        sv_file = torch.load(config['pretrained'])
        model = models.make(
            sv_file['model'], load_sd=True).to(device)
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
    else:
        model = models.make(config['model']).to(device)
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

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

def train(train_loader, model, optimizer, gradient_accumulation_steps):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

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
        pred = model(inp, batch['coord'], batch['cell'])            

        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if step % gradient_accumulation_steps == 0:
            train_loss.add(loss.item())
            optimizer.step()
            optimizer.zero_grad()

            pred = None; loss = None

        if 'scale' in batch.keys() and 'scale_max' in batch.keys():
            add_scales_to_dict(batch['scale'], batch['scale_max'])
            
    return train_loss.item()


def main(config_, save_path):
    global config, log, writer, device, scale_freq, scale_max_freq, eval_results
    scale_freq = {}
    scale_max_freq = {}
    eval_results = []
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

    # TODO: change this back
    num_iter_per_epoch = math.ceil(len(train_loader.dataset) / config['train_dataset']['batch_size']) # config['train_dataset']['batch_size'])
    iter_max = config['iter_max']
    epoch_max = math.ceil(iter_max / num_iter_per_epoch)
    epoch_val = math.floor(config.get('iter_val') / num_iter_per_epoch)
    epoch_save = math.floor(config.get('iter_save') / num_iter_per_epoch)
    config['multi_step_lr']['milestones'] = [math.floor(milestone / num_iter_per_epoch) for milestone in config['multi_step_lr']['milestones']]
    print('len dataset:', len(train_loader.dataset))
    print('epoch_max:', epoch_max)
    print('epoch_val', epoch_val)
    print('epoch_save', epoch_save)
    print('milestones', config['multi_step_lr'])

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    # Enable running on cpu as well
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Run on device: ", device)


    # get model, optimizer, and lr_scheduler from config
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    if n_gpus > 1:
        print("Use multiple gpus.")
        model = nn.parallel.DataParallel(model)

    max_val_v = -1e18

    gradient_accumulation_steps = config.get('gradient_accumulation_steps')

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # train epoch
        train_loss = train(train_loader, model, optimizer, gradient_accumulation_steps)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        # save weights
        torch.save(sv_file, os.path.join(save_path, 'iteration-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                       os.path.join(save_path, 'iteration-{}.pth'.format(epoch*num_iter_per_epoch)))
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

            val_res, _, _ = eval_psnr(val_loader, model_,
                                data_norm=config['data_norm'],
                                eval_type=config.get('eval_type'),
                                eval_bsize=config.get('eval_bsize'),
                                device=device, 
                                writer=writer,
                                epoch=epoch)

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)

            eval_results.append(val_res)

            if eval_results and (epoch % epoch_save == 0):
                with open(os.path.join(save_path, 'eval_results.pickle'), "wb") as f:
                    pickle.dump(eval_results, f)
            if val_res > max_val_v:
                max_val_v = val_res
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
