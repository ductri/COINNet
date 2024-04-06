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


def trainer_mbem(args,alg_options,logger):

	# Weighted Majority Voting
	naive_agg=weighted_majority_voting(alg_options,args,logger)			
	alg_options['A_est'] = np.zeros((args.M,args.K,args.K))
	
	# Next train the classifier on weighted majority voting predictions
	logger.info('Training classifier on WMV outputs')
	train_data_mbem=copy.deepcopy(alg_options['train_data'])
	train_data_mbem.annotations = naive_agg
	train_loader_mbem = DataLoader(dataset=train_data_mbem,
							  batch_size=args.batch_size,
							  num_workers=4,
							  shuffle=True,
							  drop_last=True)
	
	# Prepare data for training/validation and testing
	alg_options['train_loader_mbem'] = copy.deepcopy(train_loader_mbem)
	out = train_val(args,alg_options,logger,flag_soft_labels=True)
	
	
	naive_pred = out['best_train_soft_labels']
	alg_options['saved_f_model_mbem'] = out['final_model_f_dict']
		
	# Run the MBEM algorithm
	annotations_one_hot=alg_options['annotations_one_hot']
	annotators_per_sample =	alg_options['annotators_per_sample_mbem']
	logger.info('Computes posterior prob. dist. of the true labels given the noisy annotations')
	probs_est_labels,A_est = post_prob_DS(annotations_one_hot,naive_pred,annotators_per_sample) 
	print(probs_est_labels)
	alg_options['A_est'] = A_est 
		
	# Next train the classifier on MBEM predictions
	logger.info('Again training the classifier with the previous estimates')
	train_data_mbem2=copy.deepcopy(alg_options['train_data'])
	train_data_mbem2.annotations = probs_est_labels
	train_loader_mbem2 = DataLoader(dataset=train_data_mbem2,
							  batch_size=args.batch_size,
							  num_workers=4,
							  shuffle=False,
							  drop_last=False)	
	alg_options['train_loader_mbem']=copy.deepcopy(train_loader_mbem2)
	learning_rate_list=alg_options['learning_rate_list']
	best_val_acc=0
	for j in range(len(learning_rate_list)):
		args.learning_rate = learning_rate_list[j]
		logger.info('Training with learning_rate = '+str(args.learning_rate))
		out=train_val(args,alg_options,logger,flag_soft_labels=False)
		if out['best_val_acc'] >= best_val_acc:
			best_model = out['final_model_f_dict']
			best_val_acc=out['best_val_acc']	
			best_lr = learning_rate_list[j]
				
	# Perform testing
	logger.info('Testing with learning_rate = '+str(best_lr))
	test_acc = test(args,alg_options,logger,best_model)
	return test_acc
													  							  
							  
						

def train_val(args,alg_options,logger,flag_soft_labels):

	train_loader_mbem = alg_options['train_loader_mbem']
	val_loader = alg_options['val_loader']
	device=alg_options['device']
							  
	if args.dataset=='synthetic':
		# Instantiate the model f and the model for confusion matrices 
		hidden_layers=1
		hidden_units=10
		model_f = FCNN(args.R,args.K,hidden_layers,hidden_units)
				
		# The optimizers
		optimizer_f = optim.Adam(model_f.parameters(),lr=args.learning_rate,weight_decay=1e-5)
			
	elif args.dataset=='mnist':
		# Instantiate the model f and the model for confusion matrices 
		if  not flag_soft_labels:
			model_f = alg_options['saved_f_model_mbem']	
		else:
			model_f = Lenet()
		
		# The optimizers
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)
		
	elif args.dataset=='fmnist':
		# Instantiate the model f and the model for confusion matrices 
		if  not flag_soft_labels:
			model_f = alg_options['saved_f_model_mbem']	
		else:
			model_f = ResNet18_F(args.K)
		
		# The optimizers
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)
		
	elif args.dataset=='cifar10':
		# Instantiate the model f and the model for confusion matrices 
		if  not flag_soft_labels:
			model_f = alg_options['saved_f_model_mbem']	
		else:
			model_f = ResNet9(args.K)
		
		# The optimizers
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)
	
	elif args.dataset=='labelme':
		# Instantiate the model f and the model for confusion matrices 	
		if  not flag_soft_labels:
			model_f = alg_options['saved_f_model_mbem']	
		else:
			model_f = FCNN_DropoutNosoftmax(args.R,args.K)
		
		# The optimizers		
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=0)
	elif args.dataset=='music':
		# Instantiate the model f and the model for confusion matrices 	
		if  not flag_soft_labels:
			model_f = alg_options['saved_f_model_mbem']	
		else:
			model_f = FCNN_Dropout_BatchNorm(args.R,args.K)
		
		# The optimizers		
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=0)
	elif args.dataset=='cifar100':
		None
	else:
		logger.info('Incorrect choice for dataset')
	if torch.cuda.is_available:
		model_f = model_f.to(device)
		

		
	# Loss function
	loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')


	
	A_true = alg_options['A_true']
	flag_lr_scheduler=alg_options['flag_lr_scheduler']

	
	#Start training
	val_acc_list=[]
	train_acc_list=[]
	A_est_error_list=[]
	len_train_data=len(train_loader_mbem.dataset)
	len_val_data=len(val_loader.dataset)
	train_soft_labels=np.zeros((args.N,args.K))
	best_val_score = 0
	best_f_model = copy.deepcopy(model_f)
	factor=calculate_factor_for_determinant(args.M,args.K)
	for epoch in range(args.n_epoch):
		model_f.train()
		
		total_train_loss=0
		ce_loss=0
		n_train_acc=0
		#for i, data_t in enumerate(train_loader):
		for batch_x, batch_annotations, batch_annot_onehot, batch_annot_mask, batch_annot_list, batch_y in train_loader_mbem:
			flag=0
			if torch.cuda.is_available:
				batch_x=batch_x.to(device)
				batch_annotations=batch_annotations.to(device)
				batch_y = batch_y.to(device)
			optimizer_f.zero_grad()
			f_x = model_f.forward(batch_x.float())
			cross_entropy_loss=loss_function(f_x,batch_annotations)		
			loss = cross_entropy_loss
			total_train_loss+=loss.item()
			loss.backward()
			optimizer_f.step()


			# Training error
			y_hat = torch.max(F.softmax(f_x,dim=1),1)[1]
			u = (y_hat == batch_y).sum()
			n_train_acc += u.item()	

		# Validation error
		with torch.no_grad():
			model_f.eval()
			n_val_acc=0
			for batch_x,batch_y in val_loader:
				if torch.cuda.is_available:
					batch_x=batch_x.to(device)
					batch_y = batch_y.to(device)
				f_x = model_f(batch_x.float())
				y_hat = torch.max(F.softmax(f_x,dim=1),1)[1]
				u = (y_hat == batch_y).sum()
				n_val_acc += u.item()
			val_acc_list.append(n_val_acc /len_val_data )
			train_acc_list.append(n_train_acc/len_train_data)
				
			# A_est error
			A_est = alg_options['A_est'] 
			A_est_error = get_estimation_error(A_est,A_true)
			A_est_error_list.append(A_est_error)

		logger.info('epoch:{}, Total train loss: {:.4f}, ' \
				'Train Acc: {:.4f},  Val. Acc: {:.4f}, ' \
				' Estim. error: {:.4f}'\
				.format(epoch+1, total_train_loss / len_train_data*args.batch_size, \
				n_train_acc / len_train_data,n_val_acc / len_val_data,\
				A_est_error))
				
		if val_acc_list[epoch] >= best_val_score:	
			best_val_score = val_acc_list[epoch]
			best_f_model = copy.deepcopy(model_f)
			
		if flag_soft_labels:	
			if val_acc_list[epoch] >= best_val_score:	
				best_val_score = val_acc_list[epoch]
				train_soft_labels=np.zeros((args.N,args.K))
				with torch.no_grad():
					model_f.eval()
					n_test_acc=0
					for i, data in enumerate(train_loader_mbem,0):
						batch_x, batch_annotations, batch_annot_onehot, batch_annot_mask, batch_annot_list, batch_y = data
						if torch.cuda.is_available:
							batch_x=batch_x.to(device)
							batch_y = batch_y.to(device)
						f_x = F.softmax(model_f(batch_x.float()),dim=1)
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
			y_hat = torch.max(f_x,1)[1]
			u = (y_hat == batch_y).sum()
			n_test_acc += u.item()
	logger.info('Final test accuracy : {:.4f}'.format(n_test_acc/len_test_data))
	return (n_test_acc/len_test_data)
	
	
def weighted_majority_voting(alg_options,args,logger):
	logger.info('Weighted Majority Voting')
	annotator_train_label = alg_options['annotator_softmax_label_mbem']
	annotators_per_sample =	alg_options['annotators_per_sample_mbem']
	# First performing the weighted majority voting
	naive_agg = np.zeros((args.N,args.K))
	weights   = np.zeros((args.N,1))
	# First compute the weights
	for n in range(len(annotators_per_sample)):
		weights[n] = np.shape(annotators_per_sample[n])[0]
	
	weights[weights==0]=np.finfo(float).eps
	for r in range(args.M):
		naive_agg = naive_agg + (1/weights)*annotator_train_label['softmax'+ str(r) +'_label']
	return naive_agg
	
def post_prob_DS(resp_org,e_class,workers_this_example):
	# computes posterior probability distribution of the true label given the noisy labels annotated by the workers
	# and model prediction
	n = resp_org.shape[0]
	m = resp_org.shape[1]
	k = resp_org.shape[2]
	#repeat = workers_this_example.shape[1]
	
	temp_class = np.zeros((n,k))
	e_conf = np.zeros((m,k,k))
	temp_conf = np.zeros((m,k,k))
	
	#Estimating confusion matrices of each worker by assuming model prediction "e_class" is the ground truth label
	for i in range(n):
		for j in workers_this_example[i]: #range(m)
			temp_conf[j,:,:] = temp_conf[j,:,:] + np.outer(e_class[i],resp_org[i,j])
	#regularizing confusion matrices to avoid numerical issues
	for j in range(m): 
		for r in range(k):
			if (np.sum(temp_conf[j,r,:]) ==0):
				# assuming worker is spammer for the particular class if there is no estimation for that class for that worker
				temp_conf[j,r,:] = 1/k
			else:
				# assuming there is a non-zero probability of each worker assigning labels for all the classes
				temp_conf[j,r,:][temp_conf[j,r,:]==0] = 1e-10
		e_conf[j,:,:] = np.divide(temp_conf[j,:,:],np.outer(np.sum(temp_conf[j,:,:],axis =1),np.ones(k)))
	# Estimating posterior distribution of the true labels using confusion matrices of the workers and the original
	# noisy labels annotated by the workers
	for i in range(n):
		for j in workers_this_example[i]: 
			if (np.sum(resp_org[i,j]) ==1):
				temp_class[i] = temp_class[i] + np.log(np.dot(e_conf[j,:,:],np.transpose(resp_org[i,j])))
		temp_class[i] = np.exp(temp_class[i])
		temp_class[i] = np.divide(temp_class[i],np.outer(np.sum(temp_class[i]),np.ones(k)))
		e_class[i] = temp_class[i]		   
	return e_class,e_conf
	

		