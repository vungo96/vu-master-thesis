# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

import mmedited.models.restorers.basic_restorer

from mmedit.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmedit.core.distributed_wrapper import DistributedDataParallelWrapper
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model

import socket
in_fbcode = True if 'fb' in socket.gethostname() or 'facebook' in socket.gethostname() else False

if in_fbcode:
    from iopath.common.file_io import PathManager
    from iopath.fb.manifold import ManifoldPathHandler

    pathmgr = PathManager()
    pathmgr.register_handler(ManifoldPathHandler())
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description="mmediting tester")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument("--out", help="output result pickle file")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        type=str,
        help="path to store images and if not given, will not save image",
    )
    parser.add_argument("--tmpdir", help="tmp dir for writing some results")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    if "manifold" in args.config or not in_fbcode:
        print(f"load config files on ubuntu or from regu: {args.config}")
    else:
        args.config = osp.join(osp.dirname(osp.dirname(__file__)), args.config)
        print(f"load config files from fbcode with abstract path: {args.config}")
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    # update CLI args according to configs (specify test checkpoint path in config)
    if args.checkpoint == "None" and cfg.get("test_checkpoint_path", None) is not None:
        args.checkpoint = cfg.test_checkpoint_path
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            print("set random seed to", args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ["workers_per_gpu"] if k in cfg.data),
        **dict(samples_per_gpu=1, drop_last=False, shuffle=False, dist=distributed),
        **cfg.data.get("test_dataloader", {}),
    }

    data_loader = build_dataloader(dataset, **loader_cfg)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    args.save_image = args.save_path is not None
    empty_cache = cfg.get("empty_cache", False)

    if not distributed:
        if 'SwinIR' in args.checkpoint or 'Restormer' in args.checkpoint or 'SCUNet' in args.checkpoint:
            _ = load_checkpoint(model, args.checkpoint, map_location="cpu", revise_keys=[(r'^', 'generator.')])
        else:
            _ = load_checkpoint(model, args.checkpoint, map_location="cpu")
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(
            model, data_loader, save_path=args.save_path, save_image=args.save_image
        )
    else:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        model = DistributedDataParallelWrapper(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )

        device_id = torch.cuda.current_device()
        _ = load_checkpoint(
            model,
            args.checkpoint,
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        outputs = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            save_path=args.save_path,
            save_image=args.save_image,
            empty_cache=empty_cache,
        )

    if rank == 0 and "eval_result" in outputs[0]:
        print("")
        # print metrics
        stats = dataset.evaluate(outputs)
        for stat in stats:
            print("Eval-{}: {}".format(stat, stats[stat]))

        # save result pickle
        if args.out:
            print("writing results to {}".format(args.out))
            mmcv.dump(outputs, args.out)


if __name__ == "__main__":
    main()
