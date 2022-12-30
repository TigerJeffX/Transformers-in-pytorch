# encoding=utf8
import torch
import sys
import os
import math
import numpy as np
from ml_collections import ConfigDict
from torch.optim.lr_scheduler import LambdaLR

sys.path.append('../')
from data.image_cls_task import get_CIFAR10_data_loader, get_ILSVRC2012_data_loader
from model.vision_transformer import VisionTransformer, Loss
from tqdm import tqdm

import torch.multiprocessing as mp
import torch.distributed as dist
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

def get_args(dataset):
    args = ConfigDict()
    if dataset=='cifar10':
        args.dataset = 'cifar10'
        args.layer_num = 12
        args.img_size = 224
        args.train_batch_size = 640//4
        args.test_batch_size= 128
        args.patch_size = 16
        args.output_cls_num = 10
        args.hidden_size = 768
        args.head_num = 12
        args.ffn_hidden_size = 768 * 4
        args.pretrained_model_path = '../data/ViT-B_16.npz'
        args.warmup_steps = 400
        args.t_total = 10000
        args.epoch = 80
        args.smoothing_confidence = 0.9
        args.use_fp16 = True
        args.world_size = 4
        args.ckp_after = 400
        args.val_freq = 100
    elif dataset=='ILSVRC2012':
        args.dataset = 'ILSVRC2012'
        args.layer_num = 12
        args.img_size = 384
        args.train_batch_size = 160//4
        args.test_batch_size= 160//4
        args.patch_size = 16
        args.output_cls_num = 1000
        args.hidden_size = 768
        args.head_num = 12
        args.ffn_hidden_size = 768 * 4
        args.pretrained_model_path = '../data/ViT-B_16_384.npz'
        args.warmup_steps = 500
        args.t_total = 20000
        args.epoch = 3
        args.smoothing_confidence = 0.9
        args.use_fp16 = True
        args.world_size = 4
        args.ckp_after = 2000
        args.val_freq = 1000
    else:
        raise ValueError("dataset not in [cifar10, ILSVRC2012] but %s"%dataset)
    return args

def get_data_loader(args, device_id):
    if args.dataset=='cifar10':
        train_loader, test_loader = get_CIFAR10_data_loader(args, device_id)
    elif args.dataset=='ILSVRC2012':
        train_loader, test_loader = get_ILSVRC2012_data_loader(args, device_id)
    else:
        raise ValueError("dataset not in [cifar10, ILSVRC2012] but %s"%dataset)
    return train_loader, test_loader

def get_model(args):
    pretrained_model = None
    if args.pretrained_model_path:
        pretrained_model = np.load(args.pretrained_model_path)
    model = VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        channel_num=3,
        hidden_size=args.hidden_size,
        layer_num=args.layer_num,
        head_num=args.head_num,
        ffn_hidden_size=args.ffn_hidden_size,
        output_cls_num=args.output_cls_num,
        pretrained_model=pretrained_model
    )
    return model

def get_lr_scheduler(opt, args):

    class WarmupCosineSchedule(LambdaLR):

        def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
            self.warmup_steps = warmup_steps
            self.t_total = t_total
            self.cycles = cycles
            super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

        def lr_lambda(self, step):
            if step < self.warmup_steps:
                return float(step) / float(max(1.0, self.warmup_steps))
            progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
            return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
    lr_scheduler = WarmupCosineSchedule(opt, warmup_steps=args.warmup_steps, t_total=args.t_total)

    return lr_scheduler

def do_eval(model, test_data_loader, cret, args, rank):
    eval_progress = tqdm(
        test_data_loader,
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
        disable=rank not in [0]
    )
    model.eval()
    eval_loss_total = .0
    eval_sample_total = 0
    eval_sample_correct = 0
    # accelerate inference by ddp
    # need gather different process's result by dist.all_gather
    for eval_step, eval_batch in enumerate(eval_progress):
        with torch.no_grad():
            eval_x, eval_y_true = eval_batch[0].cuda(rank), eval_batch[1].cuda(rank)
            eval_y_pred = model.forward(eval_x)
            # gather all pred
            batch_eval_y_pred = [torch.zeros_like(eval_y_pred) for _ in range(args.world_size)]
            batch_eval_y_true = [torch.zeros_like(eval_y_true) for _ in range(args.world_size)]
            dist.all_gather(batch_eval_y_pred, eval_y_pred)
            dist.all_gather(batch_eval_y_true, eval_y_true)
            batch_eval_y_pred = torch.cat(batch_eval_y_pred, dim=0)
            batch_eval_y_true = torch.cat(batch_eval_y_true, dim=0)
            # batch loss and acc
            eval_loss = cret(batch_eval_y_pred, batch_eval_y_true)
            eval_sample_total += batch_eval_y_pred.shape[0]
            eval_loss_total += eval_loss.item()*batch_eval_y_pred.shape[0]
            _true = batch_eval_y_true.detach().cpu().numpy()
            _pred = torch.argmax(batch_eval_y_pred, dim=-1).detach().cpu().numpy()
            eval_sample_correct += (_true==_pred).sum()
            avg_eval_loss = eval_loss_total/eval_sample_total
            avg_eval_acc = eval_sample_correct/eval_sample_total
            # eval progress info
            eval_progress.set_description("Valid (Loss=%2.4f, Acc=%2.4f) "%(avg_eval_loss, avg_eval_acc))
    eval_loss = eval_loss_total/eval_sample_total
    eval_acc = eval_sample_correct/eval_sample_total
    return eval_loss, eval_acc

def do_train(device_id, args):

    rank = device_id

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    torch.cuda.set_device(rank)

    model = get_model(args).cuda(rank)
    opt = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0)

    model, opt = amp.initialize(model, opt, opt_level='O1')
    model = DDP(model, gradient_predivide_factor=args.world_size)

    train_loader, test_loader = get_data_loader(args, rank)
    cret = Loss(smoothing_confidence=args.smoothing_confidence, num_cls=args.output_cls_num)
    cret = cret.cuda(rank)
    lr_scheduler = get_lr_scheduler(opt, args)

    total_step = 0
    best_acc = 0.0
    best_step = total_step
    for epoch in range(args.epoch):
        train_epoch_progress = tqdm(
            train_loader,
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
            disable=rank not in [0]
        )
        train_sample_total = 0
        train_loss_total= 0
        train_loader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_epoch_progress):
            model.train()
            x, y_true = batch[0].cuda(rank), batch[1].cuda(rank)
            y_pred = model.forward(x)
            loss = cret(y_pred, y_true)
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(opt), 1.0)
            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            # update train progress info (given that data in different distribute samplers are equally divided)
            train_sample_total += x.shape[0]
            train_loss_total += loss.item()*x.shape[0]
            avg_train_loss = train_loss_total/train_sample_total
            last_lr = lr_scheduler.get_last_lr()[0]
            train_epoch_progress.set_description(
                "Train (Epoch=%s, Step=%s, Loss=%2.4f, lr=%1.6f) "%(epoch, step, avg_train_loss, last_lr))
            total_step += 1
            dist.barrier()
            if total_step>=args.ckp_after and total_step%args.val_freq==0:
                _, cur_acc = do_eval(model, test_loader, cret, args, rank)
                if cur_acc>best_acc and rank==0:
                    best_acc = cur_acc
                    best_step = total_step
                    print('best acc %s at step %s'%(best_acc, best_step))
    print('best acc %s at step %s'%(best_acc, best_step))

if __name__ == '__main__':

    dataset = sys.argv[1]

    assert dataset in ['cifar10', 'ILSVRC2012'], "dataset must in ['cifar10', 'ILSVRC2012'] but %s"%dataset

    args = get_args(dataset)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8096"

    mp.spawn(do_train, nprocs=args.world_size, args=(args,))

