import argparse
import os
import re

import yaml
import torch
import pickle
from torch.utils.data import DataLoader

import datasets
import models

from test import eval_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model_path')
    parser.add_argument('--scale')
    parser.add_argument('--max_scale', default='4')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--window', default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Run on device: ", device)

    print("Models from path: ", args.model_path)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)

    print("window: ", args.window)

    save_path = "test_curves/metric_lists/" + args.scale

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8, pin_memory=True)

    # create a regular expression to match the epoch numbers in the filenames
    regex = re.compile(r"iteration-(\d+)\.pth")

    # list all files in the folder and filter out only the ones that match the regular expression
    epoch_files = [f for f in os.listdir(args.model_path) if regex.match(f)]
    epoch_files = sorted(epoch_files, key=lambda x: int(x.split('-')[1].split('.')[0]))
    metric_list = {
        'psnr' : [],
        'ssim' : [],
        'lpips' : []
    }

    for file in epoch_files:

        model_spec = torch.load(
            os.path.join(args.model_path, file), map_location="cpu")['model']
        model = models.make(model_spec, load_sd=True).to(device)

        res_psnr, res_ssim, res_lpips = eval_psnr(loader, model,
                        data_norm=config.get('data_norm'),
                        eval_type=config.get('eval_type'),
                        eval_bsize=config.get('eval_bsize'),
                        verbose=True,
                        device=device,
                        writer=None,
                        out_dir=args.out_dir,
                        max_scale=int(args.max_scale),
                        window_size=int(args.window),
                        tag=file
                        )
        print('result psnr: {:.4f}'.format(res_psnr))
        print('result ssim: {:.4f}'.format(res_ssim))
        print('result lpips: {:.4f}'.format(res_lpips))

        metric_list['psnr'].append(res_psnr)
        metric_list['ssim'].append(res_ssim)
        metric_list['lpips'].append(res_lpips)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, 'eval_results' + args.tag + '.pickle'), "wb") as f:
        pickle.dump(metric_list, f)

    
