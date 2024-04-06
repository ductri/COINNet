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
import wandb as wandb_run


def trainer_proposed(args,alg_options,logger):

    # if args.flag_wandb:
    #     run=wandb.init(project="InstanceDependentCrowdsourcing",name="run_"+str(args.session_id),config=args)
    #     alg_options['wandb_run']=run
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
    # if args.flag_wandb:
    #     wandb_run=alg_options['wandb_run']
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
        scheduler_f = MultiStepLR(optimizer_f, milestones=alg_options['milestones'], gamma=0.1)

    elif args.dataset=='music':
        # Instantiate the model f and the model for confusion matrices
        #model_f = CrowdLayer(args.R,args.M,args.K,'fcnn_dropout_batchnorm')
        if args.proposed_init_type=='mle_based':
            A_init=confusion_matrix_init_mle_based(annotations_list,args.M,args.K)
        elif args.proposed_init_type=='identity':
            A_init=[]
        else:
            A_init=[]
        args.classifier_NN= 'fcnn_dropout_batchnorm'
        model_f = CrowdNetwork(args,A_init)

        # The optimizers
        optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler_f = MultiStepLR(optimizer_f, milestones=alg_options['milestones'], gamma=0.1)
    elif args.dataset=='cifar10' or args.dataset=='cifar100' or args.dataset=='cifar10n':
        if args.proposed_init_type=='mle_based':
            A_init=confusion_matrix_init_mle_based(annotations_list,args.M,args.K)
        elif args.proposed_init_type=='identity':
            A_init=[]
        else:
            A_init=[]

        if args.flag_two_optimizers==1:
            model_f = ClassifierNetwork(args.R,args.K,args.classifier_NN)
            model_conf = ConfusionNetwork(args.M,args.K,args.proposed_init_type,A_init,args.flag_instance_dep_modeling)
            optimizer_conf = optim.SGD(model_conf.parameters(), lr=args.learning_rate, weight_decay=0,momentum=0.9)
            optimizer_f = optim.SGD(model_f.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,momentum=0.9)
            scheduler_f = MultiStepLR(optimizer_f, milestones=alg_options['milestones'], gamma=0.1)
            scheduler_conf = MultiStepLR(optimizer_conf, milestones=alg_options['milestones'], gamma=0.1)
        else:
            model_f = CrowdNetwork(args,A_init)
            optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)

            #scheduler_f = MultiStepLR(optimizer_f, milestones=alg_options['milestones'], gamma=0.1)
            scheduler_f = optim.lr_scheduler.OneCycleLR(optimizer_f, args.learning_rate, epochs=args.n_epoch, steps_per_epoch=len(train_loader))
        # Instantiate the model f and the model for confusion matrices
        # The optimizers
        #optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        #scheduler_f = MultiStepLR(optimizer_f, milestones=alg_options['milestones'], gamma=0.1)
    else:
        logger.info('Incorrect choice for dataset')
    if torch.cuda.is_available:
        model_f = model_f.to(device)
        if args.flag_two_optimizers==1:
            model_conf = model_conf.to(device)

    # Loss function
    loss_function = torch.nn.NLLLoss(ignore_index=-1, reduction='none')

    # Magic
    #wandb_run.watch(model_f, log = "all", log_freq=100)


    # A_true = alg_options['A_true']
    flag_lr_scheduler=alg_options['flag_lr_scheduler']


    method=alg_options['method']
    if args.vol_reg_type=='max_logdeth':
        reg_loss_function =regularization_loss_logdeth
        log_file_identifier ='_logdeth'
    elif args.vol_reg_type=='min_logdetw':
        reg_loss_function=regularization_loss_logdetw
        #reg_loss_function=regularization_loss_logdeth_min
        log_file_identifier ='_logdetw'
    elif args.vol_reg_type=='max_logdetw':
        reg_loss_function=regularization_loss_logdetw_max
        #reg_loss_function=regularization_loss_logdeth_min
        log_file_identifier ='_logdetw'
    else:
        print('Invalid reg loss function!!!!!!!')



    if args.flag_warmup:
        for epoch in range(args.warmup_epoch):
            logger.info('_________________________________________________________________')
            logger.info('epoch[{}], Warmup:'.format(epoch+ 1))
            warmup(train_loader,model_f,optimizer_f,loss_function,args)


    #Start training
    val_acc_list=[]
    train_acc_list=[]
    A_est_error_list=[]
    len_train_data=len(train_loader.dataset)
    len_val_data=len(val_loader.dataset)
    train_soft_labels=np.zeros((args.N,args.K))
    best_val_score = 0
    AUC_score_instance_indep_all_data_epoch = -1*np.ones((len(train_loader.dataset),args.M,args.n_epoch))
    E_score_instance_indep_all_data_epoch = -1*np.ones((len(train_loader.dataset),args.M,args.n_epoch))
    indices_all_data_epoch = -1*np.ones((len(train_loader.dataset),args.n_epoch))
    best_f_model = copy.deepcopy(model_f)
    A_best=[]
    total_len=len(train_loader.dataset)
    weight_samples = np.ones((total_len,args.M))
    for epoch in range(args.n_epoch):
        model_f.train()
        if args.flag_two_optimizers==1:
            model_conf.train()
        total_train_loss=0
        ce_loss=0
        reg_loss=0
        reg_loss_instance_dep=0
        n_train_acc=0
        #for i, data_t in enumerate(train_loader):
        weight_samples=torch.tensor(weight_samples)
        # for ind_x, batch_x, batch_annotations, batch_annot_onehot, batch_annot_mask, batch_annot_list, batch_y,_,_,_ in train_loader:
        for batch_x, batch_annotations, ind_x, _ in train_loader:
            flag=0
            weight_samples_sel = weight_samples[ind_x,:]
            weight_samples_sel =weight_samples_sel.view(-1)
            if torch.cuda.is_available:
                batch_x=batch_x.to(device)
                batch_annotations=batch_annotations.to(device)
                # batch_y = batch_y.to(device)
                weight_samples_sel=weight_samples_sel.to(device)
            optimizer_f.zero_grad()
            if args.flag_two_optimizers==1:
                optimizer_conf.zero_grad()
                f_x, Emb_x = model_f(batch_x.float())
                if args.confusion_network_input_type == 'classifier_ouput':
                    inp = f_x
                else:
                    inp = Emb_x
                A, E = model_conf(inp)
                if args.flag_instance_dep_modeling==1:
                    Af_x = torch.einsum('ij, bikj -> ibk',f_x,(A+E))
                else:
                    Af_x = torch.einsum('ij, bkj -> ibk',f_x,A)
            else:
                f_x, Af_x,A = model_f(batch_x.float())
            Af_x = Af_x.view(-1,args.K)
            batch_annotations_mod=batch_annotations.view(-1)
            cross_entropy_loss =loss_function(Af_x.log(), batch_annotations_mod.long())
            cross_entropy_loss = torch.sum(cross_entropy_loss*weight_samples_sel)/len(torch.nonzero(weight_samples_sel))  

            if args.lam !=0:
                regularizer_loss = reg_loss_function(A,f_x,args)
            else:
                regularizer_loss=torch.tensor(0.0)
            if args.flag_instance_dep_modeling==1 and args.mu !=0:
                regularizer_loss_instance_dep = regularization_loss_error(A,f_x,E,args)
            else:
                regularizer_loss_instance_dep = torch.tensor(0)

            if(np.isnan(regularizer_loss.item()) or np.isinf(regularizer_loss.item()) or regularizer_loss.item() > 100)  :
                flag=1
                regularizer_loss=torch.tensor(0.0)
            #regularizer_loss=torch.tensor(0.0)
            loss = cross_entropy_loss+args.lam *regularizer_loss + args.mu*regularizer_loss_instance_dep
            total_train_loss+=loss.item()
            reg_loss+=regularizer_loss.item()
            reg_loss_instance_dep+=regularizer_loss_instance_dep.item()
            ce_loss+=cross_entropy_loss.item()
            loss.backward()
            optimizer_f.step()
            if args.flag_two_optimizers==1:
                optimizer_conf.step()
            if alg_options['flag_lr_scheduler']:
                scheduler_f.step()
                if args.flag_two_optimizers==1:
                    scheduler_conf.step()

            # Training error
            y_hat = torch.max(f_x,1)[1]
            # u = (y_hat == batch_y).sum()
            u = torch.zeros(1)
            n_train_acc += u.item()


        if args.flag_instance_dep_score_calc==1 and args.flag_instance_dep_modeling==1:
            AUC_score_instance_indep_all_data = np.empty((0,args.M))#np.array([]*args.M)
            indices_all_data = np.array([])
            label_count_all_data = np.array([])
            with torch.no_grad():
                model_f.eval()
                # for ind_x, batch_x, batch_annotations, _, _, _, batch_y,flag_noise_type,_,label_count in train_loader:
                for batch_x, batch_annotations, _, _ in train_loader:
                    if torch.cuda.is_available:
                        batch_x=batch_x.to(device)
                        batch_annotations=batch_annotations.to(device)
                        # batch_y = batch_y.to(device)
                        ind_x = ind_x.to(device)

                    f_x, Af_x,A = model_f(batch_x.float())
                    AUC_instance_indep_score = instance_indep_loss(A, f_x, batch_annotations,args.K)
                    AUC_score_instance_indep_all_data=np.concatenate \
                            ((AUC_score_instance_indep_all_data,AUC_instance_indep_score),  axis=0)

                    indices_all_data= \
                            np.concatenate((indices_all_data,ind_x.cpu().detach().numpy()),axis=0)

                    label_count_all_data= \
                            np.concatenate((label_count_all_data,label_count.numpy()),axis=0)



            AUC_score_instance_indep_all_data_epoch[0:len(AUC_score_instance_indep_all_data),:,epoch]= \
                    AUC_score_instance_indep_all_data
            indices_all_data_epoch[0:len(indices_all_data),epoch]= indices_all_data


            if args.instance_dep_percent_estim>0:

                if (epoch+1) % 10==0:


                    logger.info("#######Doing Sorting#########################################################")
                    AUC_score_instance_indep_all_data_sum = np.sum(AUC_score_instance_indep_all_data,axis=1)
                    AUC_score_instance_indep_all_data_sum = AUC_score_instance_indep_all_data_sum/label_count_all_data
                    indices_instance_indep_sorted=np.argsort(AUC_score_instance_indep_all_data_sum)
                    indices_all_data_sorted = indices_all_data[indices_instance_indep_sorted]
                    indices_all_data_sorted=indices_all_data_sorted.astype('int')
                    len_indices_sorted = len(indices_all_data_sorted)
                    len_selected=int(len_indices_sorted*(1-args.instance_dep_percent_estim))
                    indices_selected = indices_all_data_sorted[0:len_selected]
                    #indices_unselected = indices_all_data_sorted[len_selected+1:]
                    weight_samples = np.zeros((total_len,args.M))
                    indices_selected=indices_selected.astype('int')
                    weight_samples[indices_selected,:]=1
                    #indices_unselected_10 = indices_all_data_sorted[-10:]
                    #logger.info("indices selected")
                    #logger.info(indices_selected[0:10])
                    #np.savetxt('indices_selected.txt', indices_selected, fmt='%d')
                    #np.savetxt('indices_not_selected.txt', indices_unselected, fmt='%d')
                    #logger.info("indices not selected")
                    #logger.info(indices_unselected_10)


        # Validation error
        with torch.no_grad():
            model_f.eval()
            n_val_acc=0
            for batch_x,batch_y in val_loader:
                if torch.cuda.is_available:
                    batch_x=batch_x.to(device)
                    batch_y = batch_y.to(device)
                f_x= model_f(batch_x.float())
                y_hat = torch.max(f_x[0],1)[1]
                u = (y_hat == batch_y).sum()
                n_val_acc += u.item()
            val_acc_list.append(n_val_acc /len_val_data )
            train_acc_list.append(n_train_acc/len_train_data)

            # A_est error
            A_est = A
            A_est = A_est.detach().cpu().numpy()
            #print(A_true)
            #print(A_est)
            # A_est_error = get_estimation_error(A_est,A_true)
            A_est_error = 0.
            A_est_error_list.append(A_est_error)

        logger.info('epoch:{}, Total train loss: {:.4f}, ' \
                'CE loss: {:.4f}, Regularizer loss (Vol): {:.4f}, Regularizer loss (Instance Dep): {:.4f},'  \
                'Train Acc: {:.4f},  Val. Acc: {:.4f}, ' \
                ' Estim. error: {:.4f}'\
                .format(epoch+1, total_train_loss / len_train_data*args.batch_size, \
                ce_loss / len_train_data*args.batch_size,reg_loss / len_train_data*args.batch_size,reg_loss_instance_dep / len_train_data*args.batch_size,\
                n_train_acc / len_train_data,n_val_acc / len_val_data,\
                A_est_error))
        # if args.flag_wandb:

        wandb_run.log({"Total train loss": total_train_loss / len_train_data*args.batch_size,
            "CE loss":ce_loss / len_train_data*args.batch_size,
            "Regularizer loss (Vol)":reg_loss / len_train_data*args.batch_size,
            "Regularizer loss (Instance Dep)":reg_loss_instance_dep / len_train_data*args.batch_size,
            "Train Acc":n_train_acc / len_train_data,
            "Val. Acc":n_val_acc / len_val_data,
            "A_est_error":A_est_error})
        #wandb_run.log({"E_score_instance_indep_all_data": E_score_instance_indep_all_data,
        #             "AUC_score_instance_indep_all_data": AUC_score_instance_indep_all_data})

        if val_acc_list[epoch] >= best_val_score:
            best_val_score = val_acc_list[epoch]
            best_f_model = copy.deepcopy(model_f)
            A_best=A_est


    if args.flag_instance_dep_score_calc==1 and args.flag_instance_dep_modeling==1:
        np.save(args.log_folder+'AUC_score_instance_indep_'+str(args.session_id)+'.npy',AUC_score_instance_indep_all_data_epoch)
        np.save(args.log_folder+'indices_data'+str(args.session_id)+'.npy',indices_all_data_epoch)




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
    # if args.flag_wandb:
    wandb_run.summary["epoch_best_val_score"] = epoch_best_val_score
    wandb_run.summary["best_train_acc"] = train_acc_list[epoch_best_val_score]
    wandb_run.summary["best_val_acc"] = val_acc_list[epoch_best_val_score]
    # print(A_est)
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
            f_x = model(batch_x.float())
            y_hat = torch.max(f_x[0],1)[1]
            u = (y_hat == batch_y).sum()
            n_test_acc += u.item()
    logger.info('Final test accuracy : {:.4f}'.format(n_test_acc/len_test_data))
    # if args.flag_wandb:
        # wandb_run=alg_options['wandb_run']
    wandb_run.summary["Final test accuracy"] = n_test_acc/len_test_data
    return (n_test_acc/len_test_data)


def regularization_loss_logdeth(A,f_x,args):
    HH = torch.mm(f_x.t(),f_x)
    regularizer_loss = -torch.log(torch.linalg.det(HH))
    return regularizer_loss

def regularization_loss_logdetw_max(A,f_x,args):
    A=torch.clip(A, min=1e-12,max=1-1e-12)
    A = F.normalize(A,p=1,dim=1)
    W = A.view(args.M*args.K,args.K)
    WW = torch.mm(W.t(),W)
    regularizer_loss = -torch.log(torch.linalg.det(WW))
    return regularizer_loss

def regularization_loss_logdetw(A,f_x,args):
    if args.lam==0:
        regularizer_loss=0
        regularizer_loss=torch.tensor(regularizer_loss)
    else:
        epsilon=1e-8
        #A=torch.clip(A, min=1e-12,max=1-1e-12)
        #A = F.normalize(A,p=1,dim=1)
        W = A.view(args.M*args.K,args.K)
        WW = torch.mm(W.t(),W)
        regularizer_loss = torch.log(torch.linalg.det(WW+epsilon*torch.eye(args.K).to(args.device)))
        #regularizer_loss = torch.log(torch.linalg.det(WW))
    return regularizer_loss

def regularization_loss_logdeth_min(A,f_x,args):
    HH = torch.mm(f_x.t(),f_x)
    #regularizer_loss = torch.log(torch.linalg.det(HH+0.0001*torch.eye(args.K).to(args.device)))
    regularizer_loss = torch.log(torch.linalg.det(HH+0.001*torch.eye(args.K).to(args.device)))
    return regularizer_loss

def regularization_loss_error(A,f_x,E,args):
    if args.mu==0:
        reg_loss=0
        reg_loss=torch.tensor(reg_loss)
    else:
        epsilon=1e-8
        #E_resize = E.view(-1,E.size(1)*E.size(2))
        E_resize = E.view(-1,E.size(1)*E.size(2)*E.size(3))
        reg_loss = torch.mean((torch.sum((E_resize) ** 2, dim = 1)+epsilon)**(args.p/2.0))
        #print(E_resize[14,:])

    return reg_loss

def warmup(warmup_loader,model,optimizer,loss_function,args):
    train_loss = 0.
    train_acc = 0.
    data_num = 0.
    device=args.device
    for batch_x, batch_annotations, batch_annot_onehot, batch_annot_mask, batch_annot_list, batch_y,_ in warmup_loader:
        flag=0
        if torch.cuda.is_available:
            batch_x=batch_x.to(device)
            batch_annotations=batch_annotations.to(device)
            batch_y = batch_y.to(device)
        optimizer.zero_grad()
        f_x, _ = model.forward(batch_x.float())
        Af_x = torch.stack([f_x[n,:] for n in range(args.M)])
        #Af_x = Af_x.view(-1,args.K)
        batch_annotations=batch_annotations.view(-1)
        ce_loss =loss_function(Af_x.log(), batch_annotations.long())
        res = torch.mean(torch.sum(f_x.log() * f_x, dim=1))
        data_num = data_num + len(f_x)

        loss = ce_loss +  res

        train_loss += loss.item() * Af_x.shape[0]


        pred = torch.max(f_x, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()

        loss.backward()

        optimizer.step()
    logger.info('Warmup Loss: {:.6f},   Acc: {:.6f}'.format(
        train_loss / data_num, train_acc / data_num))

def instance_indep_loss(A,f_x, batch_annotations,K):
    with torch.no_grad():
        Af_x = torch.einsum('ij, bkj -> ikb',f_x,A)
        #Af_x = Af_x.view(-1,K)
        loss_func = torch.nn.NLLLoss(ignore_index=-1,reduction='none')
        score=loss_func(Af_x.log(), batch_annotations.long())
        #score= torch.sum(score,dim=1)
        score = score.cpu().detach().numpy()
    return score

def instance_indep_error_frob(A,f_x,E,args):
    with torch.no_grad():
        E_resize = E.view(-1,E.size(1),E.size(2)*E.size(3))
        error_frob = torch.sum((E_resize) ** 2, dim = 2)
    return error_frob.cpu().detach().numpy()

