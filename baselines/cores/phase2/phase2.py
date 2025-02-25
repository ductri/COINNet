import argparse

import itertools
import logging
import math
import os
from collections import OrderedDict
import numpy as np
import time

import torch
from torch import nn, optim
from torch.nn.functional import kl_div, softmax, log_softmax

from tqdm import tqdm
# from theconf import Config as C, ConfigArgumentParser

from .common import get_logger
from .data import get_dataloaders
from .metrics import accuracy, Accumulator
from .networks import get_model, num_class

# import hydra
from omegaconf import DictConfig, OmegaConf

from .warmup_scheduler import GradualWarmupScheduler
import constants


# @hydra.main(version_base=None, config_path="confs", config_name="resnet34_ins_0.6")
def main(conf):

    def run_epoch(model, loader_s, loader_u, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, unsupervised=False, scheduler=None):
        #tqdm_disable = bool(os.environ.get('TASK_NAME', ''))
        tqdm_disable = conf['tqdm_disable']
        tqdm_disable = False
        if verbose:
            loader_s = tqdm(loader_s, disable=tqdm_disable)
            loader_s.set_description('[%s %04d/%04d]' % (desc_default, epoch, conf['epoch']))

        iter_u = iter(loader_u)

        metrics = Accumulator()
        cnt = 0
        total_steps = len(loader_s)
        steps = 0
        #for data, label, indices in loader_s:
        for data, label in loader_s:
            steps += 1
            '''
            print('labels',label)
            np.save('labels', label)
            np.save('indices', indices)
            print('indices', indices)
            raise SystemExit
            '''
            if not unsupervised:
                data, label = data.cuda(), label.cuda()
                preds = model(data)
                loss = loss_fn(preds, label)  # loss for supervised learning
            else:
                label = label.cuda()
                try:
                    unlabel1, unlabel2 = next(iter_u)
                except StopIteration:
                    iter_u = iter(loader_u)
                    unlabel1, unlabel2 = next(iter_u)
                data_all = torch.cat([data, unlabel1, unlabel2]).cuda()
                #print('DEBUG: data', len(data))

                preds_all = model(data_all)
                preds = preds_all[:len(data)]

                loss = loss_fn(preds, label)  # loss for supervised learning
                metrics.add_dict({'sup_loss': loss.item() * len(data)})


                preds_unsup = preds_all[len(data):]
                preds1, preds2 = torch.chunk(preds_unsup, 2)
                preds1 = softmax(preds1, dim=1).detach()
                preds2 = log_softmax(preds2, dim=1)
                assert len(preds1) == len(preds2) == conf['batch_unsup']

                loss_kldiv = kl_div(preds2, preds1, reduction='none')    # loss for unsupervised
                loss_kldiv = torch.sum(loss_kldiv, dim=1)
                assert len(loss_kldiv) == len(unlabel1)
                # loss += (epoch / 200. * conf['ratio_unsup']) * torch.mean(loss_kldiv)
                if conf['ratio_mode'] == 'constant':
                    loss += conf['ratio_unsup'] * torch.mean(loss_kldiv)
                elif conf['ratio_mode'] == 'gradual':
                    loss += (epoch / float(conf['epoch'])) * conf['ratio_unsup'] * torch.mean(loss_kldiv)
                else:
                    raise ValueError


            if optimizer:
                loss.backward()
                if conf['optimizer'].get('clip', 5) > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), conf['optimizer'].get('clip', 5))

                optimizer.step()
                optimizer.zero_grad()

            top1, top5 = accuracy(preds, label, (1, 5))

            metrics.add_dict({
                'loss': loss.item() * len(data),
                'top1': top1.item() * len(data),
                'top5': top5.item() * len(data),
                })
            cnt += len(data)
            if verbose:
                postfix = metrics / cnt
                if optimizer:
                    postfix['lr'] = optimizer.param_groups[0]['lr']
                loader_s.set_postfix(postfix)

            if scheduler is not None:
                scheduler.step(epoch - 1 + float(steps) / total_steps)

            del preds, loss, top1, top5, data, label

        if tqdm_disable:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch, conf['epoch'], metrics / cnt)

        metrics /= cnt
        if optimizer:
            metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
        if verbose:
            for key, value in metrics.items():
                writer.add_scalar(key, value, epoch)
        return metrics


    def train_and_eval(tag, dataroot, metric='last', resume=False, save_path=None, only_eval=False, unsupervised=False, devices=None):
        max_epoch = conf['epoch']
        unsup_idx = conf['unsup_idx']
        if os.path.exists(unsup_idx):
            unsup_idx = np.load(unsup_idx).tolist()
            print('Unsup idx:', len(unsup_idx))
            trainloader, unsuploader, testloader = get_dataloaders(conf['dataset'], conf['batch'], conf['batch_unsup'],
                    dataroot, with_noise=True,
                    random_state=conf['random_state'], unsup_idx=unsup_idx, conf=conf)
        else:
            trainloader, unsuploader, testloader = get_dataloaders(conf['dataset'], conf['batch'], conf['batch_unsup'],
                    dataroot, with_noise=False,
                    random_state=conf['random_state'], conf=conf)

        # create a model & an optimizer
        model = get_model(conf['model'], num_class(conf['dataset']), data_parallel=True, devices=devices)

        criterion = nn.CrossEntropyLoss()
        if conf['optimizer']['type'] == 'sgd':
            optimizer = optim.SGD(
                    model.parameters(),
                    lr=conf['lr'],
                    momentum=conf['optimizer'].get('momentum', 0.9),
                    weight_decay=conf['optimizer']['decay'],
                    nesterov=conf['optimizer']['nesterov']
                    )
        else:
            raise ValueError('invalid optimizer type=%s' % conf['optimizer']['type'])

        lr_scheduler_type = conf['lr_schedule'].get('type', 'cosine')
        if lr_scheduler_type == 'cosine':
            #t_max = 600
            #print('Temp Fix for AnnealingCosine, Tmax=',t_max)
            t_max = conf['epoch']
            if conf['lr_schedule'].get('warmup', None):
                t_max -= conf['lr_schedule']['warmup']['epoch']
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.)
        else:
            raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

        if conf['lr_schedule'].get('warmup', None):
            scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=conf['lr_schedule']['warmup']['multiplier'],
                    total_epoch=conf['lr_schedule']['warmup']['epoch'],
                    after_scheduler=scheduler
                    )

        if not tag.strip():
            from .metrics import SummaryWriterDummy as SummaryWriter
            logger.warning('tag not provided, no tensorboard log.')
        else:
            from tensorboardX import SummaryWriter
        writers = [SummaryWriter(logdir='./logs/%s/%s' % (tag, x)) for x in ['train', 'test']]

        result = OrderedDict()
        epoch_start = 1

        if (resume or only_eval) and save_path and os.path.exists(save_path):
            print('Resuming from last epoch:', save_path)
            ckpt = torch.load(save_path)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            epoch_start = ckpt['epoch']

        elif os.path.exists(conf['pretrain']):
            print('Loading pretrain from:', conf['pretrain'])
            ckpt = torch.load(conf['pretrain'])
            try:
                model.load_state_dict(ckpt['state_dict'])
            except RuntimeError:
                new_state_dict = {'module.'+k: v for k, v in ckpt['state_dict'].items()}
                model.load_state_dict(new_state_dict)
        else:
            print('DEBUG: No pretrain available')

        if only_eval:
            logger.info('evaluation only+')
            model.eval()
            rs = dict()
            rs['test'] = run_epoch(model, testloader, unsuploader, criterion, None, desc_default='*test', epoch=epoch_start, writer=writers[1])
            for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'test']):
                result['%s_%s' % (key, setname)] = rs[setname][key]
            result['epoch'] = 0
            return result

        # train loop
        # global best_valid_top1
        nonlocal best_valid_top1
        best_valid_loss = 10e10
        for epoch in range(epoch_start, max_epoch + 1):
            model.train()
            rs = dict()
            rs['train'] = run_epoch(model, trainloader, unsuploader, criterion, optimizer, desc_default='train', epoch=epoch, writer=writers[0], verbose=False, unsupervised=unsupervised, scheduler=scheduler)
            print('Train At Epoch {}'.format(epoch), rs['train'])
            if math.isnan(rs['train']['loss']):
                raise Exception('train loss is NaN.')

            model.eval()
            if epoch % (10 if 'cifar' in conf['dataset'] else 30) == 0 or epoch == max_epoch:
                rs['test'] = run_epoch(model, testloader, unsuploader, criterion, None, desc_default='*test', epoch=epoch, writer=writers[1], verbose=False)
                print('Test At Epoch {}'.format(epoch), rs['test'])
                if best_valid_top1 < rs['test']['top1']:
                    best_valid_top1 = rs['test']['top1']

                if metric == 'last' or rs[metric]['loss'] < best_valid_loss:    # TODO
                    if metric != 'last':
                        best_valid_loss = rs[metric]['loss']
                    for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'test']):
                        result['%s_%s' % (key, setname)] = rs[setname][key]
                    result['epoch'] = epoch

                    writers[1].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                # save checkpoint
                if save_path:
                    logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'test': rs['test'].get_dict(),
                            },
                        'optimizer': optimizer.state_dict(),
                        'state_dict': model.state_dict()
                        }, save_path)

        del model

        return result

    # parser = ConfigArgumentParser(conflict_handler='resolve')

    logger = get_logger('Unsupervised Data Augmentation')
    logger.setLevel(logging.INFO)

    best_valid_top1 = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default=f'{constants.ROOT}/../datasets/', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--decay', type=float, default=-1)
    parser.add_argument('--unsupervised', action='store_true', default=True)
    parser.add_argument('--only-eval', action='store_true')
    parser.add_argument('--gpu', type=int, nargs='+', default=None)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args([])
    print('Unsupervised', args.unsupervised)
    assert (args.only_eval and not args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    if args.decay > 0:
        logger.info('decay reset=%.8f' % args.decay)
        conf['optimizer']['decay'] = args.decay
    if args.save:
        logger.info('checkpoint will be saved at %s', args.save)
    print('ratio_unsup', conf['ratio_unsup'])
    print('lr', conf['lr'])
    print('batch ', conf['batch'], 'batch_unsup', conf['batch_unsup'])

    t = time.time()
    result = train_and_eval(args.tag, args.dataroot, resume=args.resume, save_path=args.save, only_eval=args.only_eval, unsupervised=args.unsupervised, devices=args.gpu)
    elapsed = time.time() - t

    logger.info('training done.')
    logger.info('model: %s' % conf['model'])
    logger.info('augmentation: %s' % conf['aug'])
    logger.info(result)
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info('best top1 error in testset: %.4f' % (1. - best_valid_top1))
