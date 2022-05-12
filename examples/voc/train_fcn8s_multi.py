#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

import torch
import yaml

import torchfcn

from train_fcn32s import get_parameters
from train_fcn32s import git_hash

from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

here = osp.dirname(osp.abspath(__file__))


def main(rank,world_size):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--resume', help='checkpoint path')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--max-iteration', type=int, default=100000, help='max iteration'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-14, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--pretrained-model',
        default=torchfcn.models.FCN16s.download(),
        help='pretrained model of FCN16s',
    )
    args = parser.parse_args()

    args.model = 'FCN8s'
    args.git_hash = git_hash()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # cuda = torch.cuda.is_available()

    # torch.manual_seed(1337)
    # if cuda:
    #     torch.cuda.manual_seed(1337)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    print(device + " is available")

    #######################################
    ## 1. default process group 생성 ***
    #######################################
    print(rank)
    dist.init_process_group(backend='nccl', 
                            init_method="file:/home/shlee/workspace/CV/pj2/pytorch-fcn/examples/voc", # tmp라는 파일이 생겼다가사라짐
                            world_size=world_size, 
                            rank=rank)
                           
    # 1. dataset

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if device else {}
    train_sampler = DistributedSampler(torchfcn.datasets.SBDClassSeg(root, split='train', transform=True))
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SBDClassSeg(root, split='train', transform=True),
        batch_size=1, shuffle=False,  sampler=train_sampler, **kwargs)
    val_sampler = DistributedSampler(torchfcn.datasets.VOC2011ClassSeg(root, split='seg11valid', transform=True))
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2011ClassSeg(
            root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False,sampler=val_sampler, **kwargs)

    # 2. model

    model = torchfcn.models.FCN8s(n_class=21)
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        fcn16s = torchfcn.models.FCN16s()
        state_dict = torch.load(args.pretrained_model)
        try:
            fcn16s.load_state_dict(state_dict)
        except RuntimeError:
            fcn16s.load_state_dict(state_dict['model_state_dict'])
        model.copy_params_from_fcn16s(fcn16s)
    if device == "cuda":
        model = model.cuda(rank)

    # 3. optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': args.lr * 2, 'weight_decay': 0},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=device,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=4000,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    # main()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4"

    world_size = torch.cuda.device_count()
    print(world_size)
    mp.spawn(main,
                nprocs=world_size,
                args=(world_size,),
                join=True)
                
