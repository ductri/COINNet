import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import wandb

from .import tools, data_load
from .models import sig_t, ResNet34
from .transformer import transform_train, transform_test,transform_target
from .data_converter import load_data
from .utils import create_dir
from utils import save_model, load_model


def super_main(conf, unique_name):

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
    parser.add_argument('--save_dir', type=str, help='dir to save model files', default='saves')
    parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'mnist')
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--noise_type', type=str, default='symmetric')
    parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
    parser.add_argument('--seed', type=int, default=np.random.randint(100000))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
    parser.add_argument('--lam', type = float, default =0.0001)
    parser.add_argument('--anchor', action='store_false')



    args = parser.parse_args([])

    args.dataset = conf.data.dataset
    args.batch_size = conf.train.batch_size
    args.n_epoch = conf.train.num_epochs
    args.lam = conf.train.lam
    args.K = conf.data.K
    args.N = conf.data.N
    args.fnet_type = conf.fnet_type
    args.classifier_NN = conf.fnet_type
    path_to_model = f'{conf.method_dir}/{unique_name}'
    # args.save_dir = path_to_model

    # np.set_printoptions(precision=2,suppress=True)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # GPU
    device = torch.device('cuda:'+ str(args.device))


    # if args.dataset == 'mnist':
    #
    #     args.n_epoch = 60
    #     num_classes = 10
    #     milestones = None
    #
    #     train_data = data_load.mnist_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
    #                                          noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type,anchor=args.anchor)
    #     val_data = data_load.mnist_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
    #                                        noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)
    #     test_data = data_load.mnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    #     model = Lenet()
    #     trans = sig_t(device, args.num_classes)
    #     optimizer_trans = optim.Adam(trans.parameters(), lr=args.lr, weight_decay=0)

    # if args.dataset == 'cifar10':
    #     args.n_epoch = 80
    #
    #     args.num_classes = 10
    #     milestones = [30,60]
    #
    #     train_data = data_load.cifar10_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
    #                                          noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, anchor=args.anchor)
    #     val_data = data_load.cifar10_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
    #                                        noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)
    #     test_data = data_load.cifar10_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    #     model = ResNet18(args.num_classes)
    #     trans = sig_t(device, args.num_classes)
    #     optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)
    #
    # if args.dataset == 'cifar100':
    #     args.init = 4.5
    #     args.n_epoch = 80
    #
    #     args.num_classes = 100
    #
    #     milestones = [30, 60]
    #
    #     # train_data = data_load.cifar100_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
    #     #                                      noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, anchor=args.anchor)
    #     # val_data = data_load.cifar100_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
    #     #                                    noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)
    #     # test_data = data_load.cifar100_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    train_data, val_data, test_data = load_data(conf)
    #
    #     model = ResNet34(args.num_classes)
    #     trans = sig_t(device, args.num_classes, init=args.init)
    #     optimizer_trans = optim.Adam(trans.parameters(), lr=args.lr, weight_decay=0)

    model = ResNet34(args.num_classes)
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.Adam(trans.parameters(), lr=args.lr, weight_decay=0)
    save_dir, model_dir, matrix_dir, logs = create_dir(args)

    print(args)



    milestones = [30, 60]
    #optimizer and StepLR
    optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler1 = MultiStepLR(optimizer_es, milestones=milestones, gamma=0.1)
    scheduler2 = MultiStepLR(optimizer_trans, milestones=milestones, gamma=0.1)


    #data_loader
    train_loader = DataLoader(dataset=train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False)

    val_loader = DataLoader(dataset=val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False)

    test_loader = DataLoader(dataset=test_data,
            batch_size=args.batch_size,
            num_workers=4,
            drop_last=False)


    loss_func_ce = F.nll_loss


    #cuda
    if torch.cuda.is_available:
        model = model.to(device)
        trans = trans.to(device)



    val_loss_list = []
    val_acc_list = []
    test_acc_list = []

    # print(train_data.t, file=logs, flush=True)


    t = trans()
    est_T = t.detach().cpu().numpy()
    print(est_T)


    # estimate_error = tools.error(est_T, train_data.t)
    #
    # print('Estimation Error: {:.2f}'.format(estimate_error), file=logs, flush=True)

    def main():


        for epoch in tqdm(range(args.n_epoch), total=args.n_epoch):

            print('epoch {}'.format(epoch + 1), file=logs,flush=True)
            model.train()
            trans.train()

            train_loss = 0.
            train_vol_loss =0.
            train_acc = 0.
            val_loss = 0.
            val_acc = 0.
            eval_loss = 0.
            eval_acc = 0.
            best_val_acc = 0.

            for batch_x, batch_y in tqdm(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer_es.zero_grad()
                optimizer_trans.zero_grad()


                clean = model(batch_x)

                t = trans()

                out = torch.mm(clean, t)

                vol_loss = t.slogdet().logabsdet

                ce_loss = loss_func_ce(out.log(), batch_y.long())
                loss = ce_loss + args.lam * vol_loss

                train_loss += loss.item()
                train_vol_loss += vol_loss.item()

                pred = torch.max(out, 1)[1]
                train_correct = (pred == batch_y).sum()
                train_acc += train_correct.item()


                loss.backward()
                optimizer_es.step()
                optimizer_trans.step()

            print('Train Loss: {:.6f}, Vol_loss: {:.6f}  Acc: {:.6f}'.format(train_loss / (len(train_data))*args.batch_size, train_vol_loss / (len(train_data))*args.batch_size, train_acc / (len(train_data))))

            scheduler1.step()
            scheduler2.step()

            with torch.no_grad():
                model.eval()
                trans.eval()
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    clean = model(batch_x)
                    t = trans()

                    out = torch.mm(clean, t)
                    loss = loss_func_ce(out.log(), batch_y.long())
                    val_loss += loss.item()
                    pred = torch.max(out, 1)[1]
                    val_correct = (pred == batch_y).sum()
                    val_acc += val_correct.item()


            print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*args.batch_size, val_acc / (len(val_data))))

            wandb.log({
                'train_loss' : train_loss / len(train_data)*args.batch_size,
                'vol_loss' : train_vol_loss / (len(train_data))*args.batch_size,
                'train_acc': train_acc / len (train_data),
                'val_loss': val_loss / (len(val_data))*args.batch_size,
                'val_acc' : val_acc / (len(val_data))
                })
            if val_acc / (len(val_data)) > best_val_acc:
                best_val_acc = val_acc / (len(val_data))
                save_model(model, path=path_to_model)



            val_loss_list.append(val_loss / (len(val_data)))
            val_acc_list.append(val_acc / (len(val_data)))
            test_acc_list.append(eval_acc / (len(test_data)))


        val_loss_array = np.array(val_loss_list)
        val_acc_array = np.array(val_acc_list)
        model_index = np.argmin(val_loss_array)
        model_index_acc = np.argmax(val_acc_array)

        matrix_path = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (model_index+1)
        # final_est_T = np.load(matrix_path)
        # final_estimate_error = tools.error(final_est_T, train_data.t)

        matrix_path_acc = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (model_index_acc+1)
        # final_est_T_acc = np.load(matrix_path_acc)
        # final_estimate_error_acc = tools.error(final_est_T_acc, train_data.t)

        print("Final test accuracy: %f" % test_acc_list[model_index])
        print("Final test accuracy acc: %f" % test_acc_list[model_index_acc])
        # print("Final estimation error loss: %f" % final_estimate_error, file=logs, flush=True)
        # print("Final estimation error loss acc: %f" % final_estimate_error_acc, file=logs, flush=True)
        print("Best epoch: %d" % model_index)
        # print(final_est_T)
        logs.close()


        load_model(model, path=path_to_model)
        eval_acc = 0.
        with torch.no_grad():
            model.eval()
            trans.eval()

            for batch_x, batch_y in tqdm(test_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                clean = model(batch_x)

                loss = loss_func_ce(clean.log(), batch_y.long())
                eval_loss += loss.item()
                pred = torch.max(clean, 1)[1]
                eval_correct = (pred == batch_y).sum()
                eval_acc += eval_correct.item()

            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)) * args.batch_size,
                eval_acc / (len(test_data))))
            wandb.summary['test_acc'] = eval_acc / (len(test_data))


            est_T = t.detach().cpu().numpy()
            # estimate_error = tools.error(est_T, train_data.t)

            matrix_path = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (epoch+1)
            np.save(matrix_path, est_T)

            # print('Estimation Error: {:.2f}'.format(estimate_error), file=logs, flush=True)
            print(est_T)

    main()

