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


def trainer_dlmv(args,alg_options,logger):
	# Majority voting
	train_data=alg_options['train_data']
	y_train=alg_options['y_train']
	annotations_one_hot=alg_options['annotations_one_hot']
	pred=majority_voting(annotations_one_hot,y_train,logger)

	logger.info('Training classifier on majority voting outputs')
	train_data.annotations = np.argmax(pred,axis=1)

	# Prepare data for training/validation and testing
	train_loader = DataLoader(dataset=train_data,
							  batch_size=args.batch_size,
							  num_workers=4,
							  shuffle=True,
							  drop_last=True)

							  
							  
	alg_options['train_loader'] = train_loader

							  
	learning_rate_list=alg_options['learning_rate_list']
	
	best_val_acc = 0
	# Perform training and validation
	for j in range(len(learning_rate_list)):
		args.learning_rate = learning_rate_list[j]
		logger.info('Training with learning_rate = '+str(args.learning_rate))
		out=train_val(args,alg_options,logger)
		if out['best_val_acc'] > best_val_acc:
			best_model = out['final_model_f_dict']
			best_val_acc=out['best_val_acc']	
			best_lr = learning_rate_list[j]
				
	# Perform testing
	logger.info('Testing with learning_rate = '+str(best_lr))
	test_acc = test(args,alg_options,logger,best_model)
	return test_acc


def train_val(args,alg_options,logger):

	train_loader = alg_options['train_loader']
	val_loader = alg_options['val_loader']
	device=args.device						  
							  
	if args.dataset=='synthetic':
		# Instantiate the model f and the model for confusion matrices 
		hidden_layers=1
		hidden_units=10
		model_f = FCNN(args.R,args.K,hidden_layers,hidden_units)
				
		# The optimizers
		optimizer_f = optim.Adam(model_f.parameters(),lr=args.learning_rate,weight_decay=1e-5)
			
	elif args.dataset=='mnist':
		# Instantiate the model f and the model for confusion matrices 
		model_f = Lenet()
		
		# The optimizers
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)
		
	elif args.dataset=='fmnist':
		# Instantiate the model f and the model for confusion matrices 
		model_f = ResNet18_F(args.K)
		
		# The optimizers
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)
	
	elif args.dataset=='labelme':
		# Instantiate the model f and the model for confusion matrices 		
		model_f = FCNN_Dropout(args.R,args.K)
		
		# The optimizers		
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=0)
	elif args.dataset=='music':
		# Instantiate the model f and the model for confusion matrices 		
		model_f = FCNN_Dropout_BatchNorm(args.R,args.K)
		
		# The optimizers		
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=0)
	elif args.dataset=='cifar10':
		if args.classifier_NN=='resnet9':
			model_f = ResNet9(args.K)
		elif args.classifier_NN=='resnet18':
			model_f = ResNet18(args.K)
		elif args.classifier_NN=='resnet34':
			model_f = ResNet34(args.K)
		else:
			print("Invalid classifier function !!!!")
		# The optimizers
		#optimizer_f = optim.SGD(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4,momentum=0.9)
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)
		scheduler_f = MultiStepLR(optimizer_f, milestones=alg_options['milestones'], gamma=0.1)
		#scheduler_f = optim.lr_scheduler.OneCycleLR(optimizer_f, args.learning_rate, epochs=args.n_epoch, steps_per_epoch=len(train_loader))
	else:
		logger.info('Incorrect choice for dataset')
	if torch.cuda.is_available:
		model_f = model_f.to(device)
		
	# Loss function
	loss_function = torch.nn.NLLLoss(ignore_index=-1, reduction='mean')


	

	
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
		n_train_acc=0
		#for i, data_t in enumerate(train_loader):
		for batch_x, batch_annotations, batch_annot_onehot, batch_annot_mask, batch_annot_list, batch_y in train_loader:
			flag=0
			if torch.cuda.is_available:
				batch_x=batch_x.to(device)
				batch_annotations=batch_annotations.to(device)
				batch_y = batch_y.to(device)
			optimizer_f.zero_grad()
			f_x = model_f.forward(batch_x.float())
			f_x = f_x.view(-1,args.K)
			batch_annotations=batch_annotations.view(-1)
			cross_entropy_loss=loss_function(f_x.log(), batch_annotations.long())	
			loss = cross_entropy_loss
			total_train_loss+=loss.item()
			loss.backward()
			optimizer_f.step()
			if alg_options['flag_lr_scheduler']:
				scheduler_f.step()

			# Training error
			y_hat = torch.max(f_x,1)[1]
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
				y_hat = torch.max(f_x,1)[1]
				u = (y_hat == batch_y).sum()
				n_val_acc += u.item()
			val_acc_list.append(n_val_acc /len_val_data )
			train_acc_list.append(n_train_acc/len_train_data)
				
			# A_est error
			A_est_error = 0


		logger.info('epoch:{}, Total train loss: {:.4f}, ' \
				'Train Acc: {:.4f},  Val. Acc: {:.4f}, ' \
				' Estim. error: {:.4f}'\
				.format(epoch+1, total_train_loss / len_train_data*args.batch_size, \
				n_train_acc / len_train_data,n_val_acc / len_val_data,\
				A_est_error))
				
		if val_acc_list[epoch] > best_val_score:	
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
	
	

	