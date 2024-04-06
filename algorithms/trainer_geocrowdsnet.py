from __future__ import division
import numpy as np
import torch
import torch.optim as optim
import logging
import torch.nn.functional as F
from helpers.functions import *
from helpers.model import *
from torch.utils.data import DataLoader
from helpers.data_load import *
from helpers.transformer import *
import os
import copy
import math
from torch.optim.lr_scheduler import MultiStepLR
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import wandb


def trainer_geocrowdsnet(args,alg_options,logger):
    lamda_list=alg_options['lamda_list']
    learning_rate_list=alg_options['learning_rate_list']

    best_val_acc = -100
    # Perform training and validation
    for l in range(len(lamda_list)):
        for j in range(len(learning_rate_list)):
            args.lam = lamda_list[l]
            args.learning_rate = learning_rate_list[j]
            logger.info('Training with lambda='+str(args.lam)+' learning_rate = '+str(args.learning_rate))
            out=train_val(args,alg_options,logger)
            if out['best_val_acc'] >= best_val_acc:
                best_model = out['final_model_f_dict']
                best_val_acc=out['best_val_acc']
                best_lam, best_lr = lamda_list[l], learning_rate_list[j]

    # Perform testing
    logger.info('Testing with lambda='+str(best_lam)+' learning_rate = '+str(best_lr))
    test_acc = test(args,alg_options,logger,best_model)
    return test_acc


def train_val(args,alg_options,logger):
    train_loader = alg_options['train_loader']
    val_loader = alg_options['val_loader']
    test_loader = alg_options['test_loader']
    device = alg_options['device']
    # annotations_list = alg_options['annotations_list_maxmig']

    if args.dataset=='synthetic':
        # Instantiate the model f and the model for confusion matrices
        hidden_layers=1
        hidden_units=10
        model_f = FCNN(args.R,args.K,hidden_layers,hidden_units)
        model_A = confusion_matrices(args.device,args.M,args.K)

        # The optimizers
        optimizer_f = optim.Adam(model_f.parameters(),lr=args.learning_rate,weight_decay=1e-5)
        optimizer_A = optim.Adam(model_A.parameters(),lr=args.learning_rate,weight_decay=1e-5)

    elif args.dataset=='mnist':
        # Instantiate the model f and the model for confusion matrices
        if args.proposed_init_type=='mle_based':
            A_init=confusion_matrix_init_mle_based(annotations_list,args.M,args.K)
        elif args.proposed_init_type=='identity':
            A_init=[]
        else:
            A_init=[]
        model_f = CrowdNetwork(args.R,args.M,args.K,'lenet',args.proposed_init_type,A_init)

        # The optimizers
        optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    elif args.dataset=='fmnist':
        # Instantiate the model f and the model for confusion matrices
        if args.proposed_init_type=='mle_based':
            A_init=confusion_matrix_init_mle_based(annotations_list,args.M,args.K)
        elif args.proposed_init_type=='identity':
            A_init=[]
        else:
            A_init=[]
        model_f = CrowdNetwork(args,A_init)

        # The optimizers
        optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    elif args.dataset=='labelme':
        # Instantiate the model f and the model for confusion matrices
        if args.proposed_init_type=='mle_based':
            A_init=confusion_matrix_init_mle_based(annotations_list,args.M,args.K)
        elif args.proposed_init_type=='identity':
            A_init=[]
        else:
            A_init=[]
        args.classifier_NN= 'fcnn_dropout'
        model_f = CrowdNetwork(args,A_init)

        # The optimizers
        optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    elif args.dataset=='music':
        # Instantiate the model f and the model for confusion matrices
        #model_f = CrowdLayer(args.R,args.M,args.K,'fcnn_dropout_batchnorm')
        if args.proposed_init_type=='mle_based':
            A_init=confusion_matrix_init_mle_based(annotations_list,args.M,args.K)
        elif args.proposed_init_type=='identity':
            A_init=[]
        else:
            A_init=[]
        args.classifier_NN= 'fcnn_dropout'
        model_f = CrowdNetwork(args,A_init)
        # The optimizers
        optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    elif args.dataset=='cifar10' or args.dataset == 'cifar10n':
        if args.proposed_init_type=='mle_based':
            A_init=confusion_matrix_init_mle_based(annotations_list,args.M,args.K)
        elif args.proposed_init_type=='identity':
            A_init=[]
        else:
            A_init=[]
        model_f = CrowdNetwork(args,A_init)
        # Instantiate the model f and the model for confusion matrices
        # The optimizers
        #optimizer_f = optim.SGD(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4,momentum=0.9)
        optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        #scheduler_f = MultiStepLR(optimizer_f, milestones=alg_options['milestones'], gamma=0.1)
        scheduler_f = optim.lr_scheduler.OneCycleLR(optimizer_f, args.learning_rate, epochs=args.n_epoch, steps_per_epoch=len(train_loader))
    else:
        logger.info('Incorrect choice for dataset')
    if torch.cuda.is_available:
        model_f = model_f.to(device)

    # Loss function
    loss_function = torch.nn.NLLLoss(ignore_index=-1, reduction='mean')




    # A_true = alg_options['A_true']
    flag_lr_scheduler=alg_options['flag_lr_scheduler']


    method=alg_options['method']
    if method=='GEOCROWDNET':
        reg_loss_function =regularization_loss_logdeth
        log_file_identifier ='_logdeth'
    elif method =='VOLMINEECS_LOGDETW':
        reg_loss_function=regularization_loss_logdetw
        #reg_loss_function=regularization_loss_logdeth_min
        log_file_identifier ='_logdetw'
    else:
        print('Invalid reg loss function!!!!!!!')


    #Start training
    val_acc_list=[]
    train_acc_list=[]
    A_est_error_list=[]
    len_train_data=len(train_loader.dataset)
    len_val_data=len(val_loader.dataset)
    train_soft_labels=np.zeros((args.N,args.K))
    best_val_score = 0
    best_f_model = copy.deepcopy(model_f)
    for epoch in range(args.n_epoch):
        model_f.train()

        total_train_loss=0
        ce_loss=0
        reg_loss=0
        n_train_acc=0
        #for i, data_t in enumerate(train_loader):
        # for _,batch_x, batch_annotations, batch_annot_onehot, batch_annot_mask, batch_annot_list, batch_y,_,_,_ in train_loader:
        for batch_x, batch_annotations, _, _ in train_loader:
            flag=0
            if torch.cuda.is_available:
                batch_x=batch_x.to(device)
                batch_annotations=batch_annotations.to(device)
                # batch_y = batch_y.to(device)
            optimizer_f.zero_grad()
            f_x, Af_x,A = model_f.forward(batch_x.float())
            Af_x = Af_x.view(-1,args.K)
            batch_annotations=batch_annotations.view(-1)
            cross_entropy_loss =loss_function(Af_x.log(), batch_annotations.long())
            regularizer_loss = reg_loss_function(A,f_x,args)
            if(np.isnan(regularizer_loss.item()) or np.isinf(regularizer_loss.item()) or regularizer_loss.item() > 100)  :
                flag=1
                regularizer_loss=torch.tensor(0.0)
            #regularizer_loss=torch.tensor(0.0)
            loss = cross_entropy_loss+args.lam *regularizer_loss
            total_train_loss+=loss.item()
            reg_loss+=regularizer_loss.item()
            ce_loss+=cross_entropy_loss.item()
            loss.backward()
            optimizer_f.step()
            if alg_options['flag_lr_scheduler']:
                scheduler_f.step()

            # Training error
            y_hat = torch.max(f_x,1)[1]
            # u = (y_hat == batch_y).sum()
            u = torch.zeros(1)
            n_train_acc += u.item()



        # Validation error
        with torch.no_grad():
            model_f.eval()
            n_val_acc=0
            for batch_x,batch_y in val_loader:
                if torch.cuda.is_available:
                    batch_x=batch_x.to(device)
                    batch_y = batch_y.to(device)
                f_x,Af_x,A = model_f(batch_x.float())
                y_hat = torch.max(f_x,1)[1]
                u = (y_hat == batch_y).sum()
                n_val_acc += u.item()
            val_acc_list.append(n_val_acc /len_val_data )
            train_acc_list.append(n_train_acc/len_train_data)

            # A_est error
            A_est = A
            A_est = A_est.detach().cpu().numpy()
            # A_est_error = get_estimation_error(A_est,A_true)
            A_est_error = 0.
            A_est_error_list.append(A_est_error)

        logger.info('epoch:{}, Total train loss: {:.4f}, ' \
                'CE loss: {:.4f}, Regularizer loss: {:.4f}, '  \
                'Train Acc: {:.4f},  Val. Acc: {:.4f}, ' \
                ' Estim. error: {:.4f}'\
                .format(epoch+1, total_train_loss / len_train_data*args.batch_size, \
                ce_loss / len_train_data*args.batch_size,reg_loss / len_train_data*args.batch_size,\
                n_train_acc / len_train_data,n_val_acc / len_val_data,\
                A_est_error))


        wandb.log({'Total train loss': total_train_loss / len_train_data*args.batch_size,  \
                'CE loss': ce_loss / len_train_data*args.batch_size,
                'Regularizer loss': reg_loss / len_train_data*args.batch_size,   \
                'Train Acc': n_train_acc / len_train_data,
                'Val. Acc': n_val_acc / len_val_data,
                'Estim. error': A_est_error
                })

        if val_acc_list[epoch] >= best_val_score:
            best_val_score = val_acc_list[epoch]
            best_f_model = copy.deepcopy(model_f)

    val_acc_array = np.array(val_acc_list)
    epoch_best_val_score = np.argmax(val_acc_array)
    logger.info("Best epoch based on validation: %d" % epoch_best_val_score)
    logger.info("Final train accuracy : %f" % train_acc_list[epoch_best_val_score])
    logger.info("Best val accuracy : %f" % val_acc_list[epoch_best_val_score])

    out={}
    out['epoch_best_val_score']= epoch_best_val_score
    out['best_train_acc']= train_acc_list[epoch_best_val_score]
    out['best_val_acc']= val_acc_list[epoch_best_val_score]
    out['best_train_soft_labels']=train_soft_labels
    out['final_model_f_dict']=best_f_model

    wandb.summary['epoch_best_val_score']= epoch_best_val_score
    wandb.summary['best_train_acc']= train_acc_list[epoch_best_val_score]
    wandb.summary['best_val_acc']= val_acc_list[epoch_best_val_score]
    wandb.summary['best_train_soft_labels']=train_soft_labels


    #if method=='VOLMINEECS_LOGDETW':
    #    for m in range(5):
    #        df_cm = pd.DataFrame(A_true[m], range(args.K), range(args.K))
    #        sn.set(font_scale=1.4) # for label size
    #        sn.heatmap(df_cm, vmin=0, vmax=1, annot=True,  annot_kws={"size": 8}) # font size
    #        plt.savefig(args.log_folder+'A_true_'+str(m)+log_file_identifier+'.eps')
    #        plt.savefig(args.log_folder+'A_true_'+str(m)+log_file_identifier+'.png')
    #        plt.close()
    #
    #        df_cm = pd.DataFrame(A_est[m], range(args.K), range(args.K))
    #        sn.set(font_scale=1.4) # for label size
    #        sn.heatmap(df_cm, vmin=0, vmax=1, annot=True,  annot_kws={"size": 8}) # font size
    #        plt.savefig(args.log_folder+'A_est_'+str(m)+log_file_identifier+'.eps')
    #        plt.savefig(args.log_folder+'A_est_'+str(m)+log_file_identifier+'.png')
    #        plt.close()
    #log_file_name = args.log_folder+'log_'+str(args.learning_rate
    return out


def test(args,alg_options,logger, best_model):
    test_loader = alg_options['test_loader']
    model=best_model
    device=alg_options['device']
    #Start testing
    n_test_acc=0
    len_test_data=len(test_loader.dataset)
    with torch.no_grad():
        model.eval()
        for batch_x,batch_y in test_loader:
            if torch.cuda.is_available:
                batch_x=batch_x.to(device)
                batch_y = batch_y.to(device)
            f_x,Af_x,A = model(batch_x.float())
            y_hat = torch.max(f_x,1)[1]
            u = (y_hat == batch_y).sum()
            n_test_acc += u.item()
    logger.info('Final test accuracy : {:.4f}'.format(n_test_acc/len_test_data))
    wandb.summary['Final test accuracy'] = n_test_acc/len_test_data
    return (n_test_acc/len_test_data)


def regularization_loss_logdeth(A,f_x,args):
    HH = torch.mm(f_x.t(),f_x)
    regularizer_loss = -torch.log(torch.linalg.det(HH))
    return regularizer_loss

def regularization_loss_logdetw(A,f_x,args):
    A=torch.clip(A, min=1e-12,max=1-1e-12)
    A = F.normalize(A,p=1,dim=1)
    W = A.view(args.M*args.K,args.K)
    WW = torch.mm(W.t(),W)
    regularizer_loss = -torch.log(torch.linalg.det(WW))
    return regularizer_loss

def regularization_loss_logdeth_min(A,f_x,args):
    HH = torch.mm(f_x.t(),f_x)
    #regularizer_loss = torch.log(torch.linalg.det(HH+0.0001*torch.eye(args.K).to(args.device)))
    regularizer_loss = torch.log(torch.linalg.det(HH+0.001*torch.eye(args.K).to(args.device)))
    return regularizer_loss

