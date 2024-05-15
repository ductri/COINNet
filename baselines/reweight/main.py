import os
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path
import wandb

from .my_data_converter import convert_train_batch_majority
from my_dataset import get_dataset
from . import tools, data_load, Lenet, Resnet
from .transformer import transform_train, transform_test,transform_target
from .loss import reweight_loss
import utils


def main(conf, unique_name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
    parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
    parser.add_argument('--model_dir', type=str, help='dir to save model files', default='model')
    parser.add_argument('--prob_dir', type=str, help='dir to save output probability files', default='prob' )
    parser.add_argument('--matrix_dir', type=str, help='dir to save estimated matrix', default='matrix')
    parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'mnist')
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--n_epoch_estimate', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--percentile', type=int, default=97)
    parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 100)
    parser.add_argument('--seed', type=int, default=np.random.randint(100000))
    parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args([])


    #seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    # if args.dataset == 'cifar10':
    #     args.n_epoch = 200
    #     args.n_epoch_estimate = 20
    #     args.num_classes = 10
    #     train_data = data_load.cifar10_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
    #             noise_rate=args.noise_rate, random_seed=args.seed)
    #     val_data = data_load.cifar10_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
    #             noise_rate=args.noise_rate, random_seed=args.seed)
    #     test_data = data_load.cifar10_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    #     estimate_state = True
    #     model = Resnet.ResNet18(args.num_classes)
    #     # optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    #     # scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
    # else:
    #     raise Exception(f'Typo dataset: {args.dataset}')
    args.n_epoch = conf.train.num_epochs
    args.n_epoch_estimate = conf.train.num_estimate_epochs
    args.num_classes = conf.data.K
    args.model_dir = f'{conf.model_dir}/{unique_name}'
    args.dataset = conf.data.dataset
    args.batch_size = conf.train.batch_size

    estimate_state = True
    model = Resnet.ResNet34(args.num_classes, conf)


    #optimizer nd StepLR
    optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

    data_module = get_dataset(conf)
    data_module.prepare_data()
    data_module.setup('fit')
    data_module.setup('test')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    estimate_loader = data_module.train_dataloader()
    train_data = train_loader.dataset
    val_data = val_loader.dataset
    test_data = test_loader.dataset

    #loss
    loss_func_reweight = reweight_loss()
    loss_func_ce = nn.CrossEntropyLoss()

    #cuda
    if torch.cuda.is_available:
        model = model.cuda()
        loss_func_reweight = loss_func_reweight.cuda()
        loss_func_ce = loss_func_ce.cuda()

    #mkdir
    model_save_dir = args.model_dir + '/' + args.dataset + '/' + 'noise_rate_%s'%(args.noise_rate)
    #print(model_save_dir)

    if not os.path.exists(model_save_dir):
        os.system('mkdir -p %s'%(model_save_dir))

    prob_save_dir = args.prob_dir + '/' + args.dataset + '/' + 'noise_rate_%s'%(args.noise_rate)

    if not os.path.exists(prob_save_dir):
        os.system('mkdir -p %s'%(prob_save_dir))

    matrix_save_dir = args.matrix_dir + '/' + args.dataset + '/' + 'noise_rate_%s'%(args.noise_rate)

    if not os.path.exists(matrix_save_dir):
        os.system('mkdir -p %s'%(matrix_save_dir))

    #estimate transition matrix
    index_num = int(len(train_data) / args.batch_size)
    A = torch.zeros((args.n_epoch_estimate, len(train_data), args.num_classes))
    val_acc_list = []
    error_list = []
    total_index =  index_num + 1

    print('Estimate transition matirx......Waiting......')

    for epoch in range(args.n_epoch_estimate):
        print('epoch {}'.format(epoch + 1))
        model.train()
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.

        for batch in train_loader:
            batch_x, batch_y = convert_train_batch_majority(batch)
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer_es.zero_grad()
            out = model(batch_x)
            loss = loss_func_ce(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer_es.step()

        torch.save(model.state_dict(), model_save_dir + '/'+ 'epoch_%d.pth'%(epoch+1))
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data))*args.batch_size, train_acc / (len(train_data))))

        with torch.no_grad():
            model.eval()
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                out = model(batch_x)
                loss = loss_func_ce(out, batch_y)
                val_loss += loss.item()
                pred = torch.max(out, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()

        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*args.batch_size, val_acc / (len(val_data))))
        val_acc_list.append(val_acc / (len(val_data)))

        with torch.no_grad():
            model.eval()
            for index,batch  in enumerate(estimate_loader):
                batch_x,batch_y = convert_train_batch_majority(batch)
                batch_x = batch_x.cuda()
                out = model(batch_x)
                out = F.softmax(out,dim=1)
                out = out.cpu()
                if index <= index_num:
                    A[epoch][index*args.batch_size:(index+1)*args.batch_size, :] = out
                else:
                    A[epoch][index_num*args.batch_size, len(train_data), :] = out

    val_acc_array = np.array(val_acc_list)
    model_index = np.argmax(val_acc_array)

    A_save_dir = prob_save_dir + '/' + 'prob.npy'
    np.save(A_save_dir, A)
    prob_ = np.load(A_save_dir)

    transition_matrix_ = tools.fit(prob_[model_index, :, :], args.num_classes, estimate_state)
    transition_matrix = tools.norm(transition_matrix_)

    matrix_path = matrix_save_dir + '/' + 'transition_matrix.npy'
    np.save(matrix_path, transition_matrix)
    T = torch.from_numpy(transition_matrix).float().cuda()

    # initial parameters

    estimate_model_path = model_save_dir + '/' + 'epoch_%s.pth'%(model_index+1)
    estimate_model_path = torch.load(estimate_model_path)
    model.load_state_dict(estimate_model_path)

    print('Estimate finish.....Training......')

    best_val_acc = 0.
    for epoch in range(args.n_epoch):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        eval_loss = 0.
        eval_acc = 0.
        scheduler.step()
        model.train()
        # for batch_x, batch_y in train_loader:
        for batch in train_loader:
            batch_x, batch_y = convert_train_batch_majority(batch)

            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer.zero_grad()
            out = model(batch_x)
            prob = F.softmax(out, dim=1)
            prob = prob.t()
            loss = loss_func_reweight(out, T, batch_y)
            out_forward = torch.matmul(T.t(), prob)
            out_forward = out_forward.t()
            train_loss += loss.item()
            pred = torch.max(out_forward, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for batch_x,batch_y in val_loader:
                model.eval()
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                out = model(batch_x)
                prob = F.softmax(out, dim=1)
                prob = prob.t()
                loss = loss_func_reweight(out, T, batch_y)
                out_forward = torch.matmul(T.t(), prob)
                out_forward = out_forward.t()
                val_loss += loss.item()
                pred = torch.max(out_forward, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()

        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data))*args.batch_size, train_acc / (len(train_data))))
        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*args.batch_size, val_acc / (len(val_data))))
        wandb.log({'train_loss': train_loss / (len(train_data))*args.batch_size,
            'train_acc': train_acc / (len(train_data)),
            'val_loss': val_loss / (len(val_data))*args.batch_size,
            'val_acc': val_acc / (len(val_data))
            })
        current_val_acc = val_acc / (len(val_data))
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            path_save = f'{conf.root}/out_reweight/{unique_name}.pt'
            utils.save_model(model, path_save)
            print(f'Save best model at {path_save} at epoch {epoch} with val_acc: {best_val_acc}')

    utils.load_model(model, path_save)
    with torch.no_grad():
        model.eval()
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            out = model(batch_x)
            loss = loss_func_ce(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            eval_correct = (pred == batch_y).sum()
            eval_acc += eval_correct.item()

        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data))*args.batch_size, eval_acc / (len(test_data))))
        wandb.summary['test_loss'] = eval_loss / (len(test_data))*args.batch_size
        wandb.summary['test_acc'] = eval_acc / (len(test_data))


