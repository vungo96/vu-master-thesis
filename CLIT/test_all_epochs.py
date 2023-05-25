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
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Run on device: ", device)

    print("Models from path: ", args.model_path)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)

    save_path = "test_curves/psnr_lists/" + args.scale

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        num_workers=8,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    # create a regular expression to match the epoch numbers in the filenames
    regex = re.compile(r"epoch-(\d+)\.pth")

    # list all files in the folder and filter out only the ones that match the regular expression
    epoch_files = [f for f in os.listdir(args.model_path) if regex.match(f)]
    epoch_files = sorted(epoch_files, key=lambda x: int(x.split('-')[1].split('.')[0]))
    psnr_list = []

    for file in epoch_files:

        model_spec = torch.load(
            os.path.join(args.model_path, file), map_location="cpu")['model']
        model = models.make(model_spec, load_sd=True).to(device)

        res = eval_psnr(
            loader,
            model,
            args.tag,
            data_norm=config.get('data_norm'),
            eval_type=config.get('eval_type'),
            eval_bsize=config.get('eval_bsize'),
            scale_max=int(args.scale_max),
            save_dir=file.split('.pth')[0],
            verbose=True
        )
        psnr_list.append(res)
        print('result: {:.4f}'.format(res))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, 'eval_results' + args.tag + '.pickle'), "wb") as f:
        pickle.dump(psnr_list, f)

    