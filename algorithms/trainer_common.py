from __future__ import division
import numpy as np
import torch
import torch.optim as optim
import logging
import torch.nn.functional as F
from functions import *
from model import *
from torch.utils.data import DataLoader
from data_load import *
from transformer import *
import os
import copy
import math
from torch.optim.lr_scheduler import MultiStepLR


def train_val(args,alg_options,logger):

    train_loader = alg_options['train_loader']
    val_loader = alg_options['val_loader']
    test_loader = alg_options['test_loader']
    model_f=alg_options['model_f']
    model_A=alg_options['model_A']
    optimizer_f=alg_options['optimizer_f']
    optimizer_A=alg_options['optimizer_A']
    scheduler_f=alg_options['scheduler_f']
    scheduler_A=alg_options['scheduler_A']
    loss_function=alg_options['loss_function']
    device=alg_options['device']
    method=alg_options['method']
    A_true = alg_options['A_true']
    flag_lr_scheduler=alg_options['flag_lr_scheduler']


    #Start training
    val_acc_list=[]
    train_acc_list=[]
    A_est_error_list=[]
    len_train_data=len(train_loader.dataset)
    len_val_data=len(val_loader.dataset)
    train_soft_labels=np.zeros((args.N,args.K))
    best_val_score = 0
    best_f_model = copy.deepcopy(model_f.state_dict())
    best_A_model = copy.deepcopy(model_A.state_dict())
    factor=calculate_factor_for_determinant(args.M,args.K)
    for epoch in range(args.n_epoch):
        model_f.train()
        if method!= 'MBEM':
            model_A.train()

        total_train_loss=0
        ce_loss=0
        reg_loss=0
        n_train_acc=0
        #for i, data_t in enumerate(train_loader):
        for batch_x,batch_annotations, batch_annotator_mask,batch_y in train_loader:
            flag=0
            if torch.cuda.is_available:
                batch_x=batch_x.to(device)
                batch_annotations=batch_annotations.to(device)
                batch_annotator_mask=batch_annotator_mask.to(device)
                batch_y = batch_y.to(device)
            optimizer_f.zero_grad()
            f_x = model_f.forward(batch_x.float())
            cross_entropy_loss=0
            if method== 'MBEM' or method== 'DL_MAJORITYVOTING' or method== 'DL_DAWID_SKENE_EM'  :
                cross_entropy_loss+=loss_function(f_x,batch_annotations)
                regularizer_loss=torch.tensor(0.0)
            elif method== 'CROWDLAYER':
                # or method == 'VOLMINEECS_LOGDETH'
                optimizer_A.zero_grad()
                Af_x = model_A.forward(f_x)
                Af_x = Af_x.view(-1,args.K)
                batch_annotations=batch_annotations.view(-1)
                cross_entropy_loss+=loss_function(Af_x,batch_annotations)
                if method == 'CROWDLAYER':
                    regularizer_loss=torch.tensor(0.0)
                else:
                    HH = torch.mm(f_x.t(),f_x)
                    regularizer_loss = -torch.log(torch.linalg.det(HH))
                    if(np.isnan(regularizer_loss.item()) or np.isinf(regularizer_loss.item())):
                        flag=1
                        regularizer_loss=torch.tensor(0.0)
            else:
                optimizer_A.zero_grad()
                A = model_A.forward()
                if torch.cuda.is_available:
                    A = A.to(device)
                Af_x = torch.einsum('ik, bkj -> ibj',f_x,A)
                Af_x = Af_x.view(-1,args.K)
                batch_annotations=batch_annotations.view(-1)
                cross_entropy_loss+=loss_function(Af_x,batch_annotations)
                if method=='VOLMINEECS_LOGDET':
                    I=torch.eye(args.K)
                    I=I.to(device)
                    W = A.view(args.M*args.K,args.K)
                    WW = torch.mm(W.t(),W)+0.001*I
                    regularizer_loss = WW.slogdet().logabsdet
                elif method=='VOLMINEECS':
                    W = A.view(args.M*args.K,args.K)
                    WW = torch.mm(W.t(),W)
                    regularizer_loss = torch.linalg.det(WW)/(10**(factor-1))
                elif method =='VOLMINEECS_LOGDETH':
                    HH = torch.mm(f_x.t(),f_x)
                    regularizer_loss = -torch.log(torch.linalg.det(HH))
                    if(np.isnan(regularizer_loss.item()) or np.isinf(regularizer_loss.item()) or regularizer_loss.item() > 20)  :
                        flag=1
                        regularizer_loss=torch.tensor(0.0)
                elif method=='TRACEREGEECS':
                    regularizer_loss=0
                    for i in range(args.M):
                        regularizer_loss+=torch.trace(A[i])
                else:
                    regularizer_loss=torch.tensor(0.0)


            loss = cross_entropy_loss+args.lam* regularizer_loss
            total_train_loss+=loss.item()
            reg_loss+=regularizer_loss.item()
            ce_loss+=cross_entropy_loss.item()

            y_hat = torch.max(f_x,1)[1]
            u = (y_hat == batch_y).sum()
            n_train_acc += u.item()
            if flag==0:
                loss.backward()
                optimizer_f.step()
                if flag_lr_scheduler:
                    scheduler_f.step()
            if method!= 'MBEM' and method!= 'DL_MAJORITYVOTING' or method!= 'DL_DAWID_SKENE_EM':
                optimizer_A.step()
                if flag_lr_scheduler:
                    scheduler_A.step()




        with torch.no_grad():
            model_f.eval()
            if method!= 'MBEM' and method!= 'DL_MAJORITYVOTING' or method!= 'DL_DAWID_SKENE_EM':
                model_A.eval()
            n_val_acc=0
            for batch_x,batch_y in val_loader:
                if torch.cuda.is_available:
                    batch_x=batch_x.to(device)
                    batch_y = batch_y.to(device)
                f_x = model_f(batch_x.float())
                y_hat = torch.max(f_x,1)[1]
                u = (y_hat == batch_y).sum()
                n_val_acc += u.item()
            val_acc_list.append(n_val_acc /len_val_data )
            train_acc_list.append(n_train_acc/len_train_data)


            if method== 'MBEM':
                A_est = alg_options['A_est']
            elif method== 'CROWDLAYER':
                #or method == 'VOLMINEECS_LOGDETH'
                A_est = model_A.A#np.zeros((args.M,args.K,args.K))
                A_est = A_est.detach().cpu().numpy()
            elif method == 'VOLMINEECS' or method == 'VOLMINEECS_LOGDET' or method == 'TRACEREGEECS' or method == 'NOREGEECS' :
                A_est = model_A()
                A_est = A_est.detach().cpu().numpy()
            else:
                A_est = np.zeros((args.M,args.K,args.K))
            A_est_error = get_estimation_error(A_est,A_true)
            A_est_error_list.append(A_est_error)

        logger.info('epoch:{}, Total train loss: {:.4f}, ' \
                'CE loss: {:.4f}, Regularizer loss: {:.4f}, '  \
                'Train Acc: {:.4f},  Val. Acc: {:.4f}, ' \
                ' Estim. error: {:.4f}'\
                .format(epoch+1, total_train_loss / len_train_data*args.batch_size, \
                ce_loss / len_train_data*args.batch_size,reg_loss / len_train_data*args.batch_size,\
                n_train_acc / len_train_data,n_val_acc / len_val_data,\
                A_est_error))

        if val_acc_list[epoch] > best_val_score:
            best_val_score = val_acc_list[epoch]
            best_f_model = copy.deepcopy(model_f.state_dict())
            best_A_model = copy.deepcopy(model_A.state_dict())

        if method== 'MBEM':
            if val_acc_list[epoch] > best_val_score:
                best_val_score = val_acc_list[epoch]
                train_soft_labels=np.zeros((args.N,args.K))
                with torch.no_grad():
                    model_f.eval()
                    n_test_acc=0
                    for i, data in enumerate(train_loader,0):
                        batch_x, batch_annotations, batch_annotator_mask, batch_y = data
                        if torch.cuda.is_available:
                            batch_x=batch_x.to(device)
                            batch_y = batch_y.to(device)
                        f_x = model_f(batch_x.float())
                        train_soft_labels[(i)*len(batch_x):(i+1)*len(batch_x),:]=f_x.detach().cpu().numpy()

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
    out['final_model_A_dict']=best_A_model
    return out


def test(args,alg_options,logger, best_model):

    test_loader = alg_options['test_loader']
    model_f=alg_options['model_f']
    model_f.load_state_dict(best_model)
    device=alg_options['device']
    #Start testing
    n_test_acc=0
    len_test_data=len(test_loader.dataset)
    method=alg_options['method']
    with torch.no_grad():
        model_f.eval()
        for batch_x,batch_y in test_loader:
            if torch.cuda.is_available:
                batch_x=batch_x.to(device)
                batch_y = batch_y.to(device)
            f_x = model_f(batch_x.float())
            y_hat = torch.max(f_x,1)[1]
            u = (y_hat == batch_y).sum()
            n_test_acc += u.item()
    logger.info('Final test accuracy : {:.4f}'.format(n_test_acc/len_test_data))
    return (n_test_acc/len_test_data)




