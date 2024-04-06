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

def trainer_doctornet(args,alg_options,logger):

	learning_rate_list=alg_options['learning_rate_list']
	
	best_val_acc = 0
	# Perform training and validation
	logger.info('Training')
	out=train_val(args,alg_options,logger)
	if out['best_val_acc'] > best_val_acc:
		best_model = out['final_model_f_dict']
		best_val_acc=out['best_val_acc']	
				
	# Perform testing
	logger.info('Testing')
	test_acc = test(args,alg_options,logger,best_model)


def train_val(args,alg_options,logger):
	train_loader = alg_options['train_loader']
	val_loader = alg_options['val_loader']
	device=args.device						  
							  
	if args.dataset=='synthetic':
		# Instantiate the model f and the model for confusion matrices 
		hidden_layers=1
		hidden_units=10
		model = FCNN(args.R,args.K,hidden_layers,hidden_units)
				

			
	elif args.dataset=='mnist':
		# Instantiate the model f and the model for confusion matrices 
		model = Lenet()
		
	
	elif args.dataset=='labelme':
		# Instantiate the model f and the model for confusion matrices		 
		model = DoctorNet(args.K, args.M, 'W',2048, None)		
		# The optimizers		
	elif args.dataset=='cifar100':
		None
	else:
		logger.info('Incorrect choice for dataset')
	if torch.cuda.is_available:
		model = model.to(device)
		

	alg_options['model']=model
	# Ignore annotators labeling which is -1
	criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

	# Freeze feature extractor
	model.annotators.requires_grad = True
	for param in model.feature_extractor.parameters():
		param.requires_grad = False
	optimizer_doctornet = optim.Adam([model.annotators], lr=1e-3, weight_decay=0)

	logger.info('Start DoctorNet training!')
	best_model = copy.deepcopy(model)
	best_accuracy = 0
	len_train_data=len(train_loader.dataset)
	len_val_data=len(val_loader.dataset)
	for epoch in range(args.n_epoch):
		train_loss = 0.0
		train_correct = 0
		model.train()
		for x, y, annotation in train_loader:
			model.zero_grad()

			x, y, annotation = x.to(args.device), y.to(args.device), annotation.to(args.device)
			pred, _ = model(x)
			
			pred = pred.view(-1, args.K)
			annotation = annotation.view(-1)
			loss = criterion(pred, annotation)

			loss.backward()
			optimizer_doctornet.step()
			train_loss += loss.item()

			pred = pred.view(-1, args.M, args.K)
			pred = torch.sum(pred, axis=1)
			pred = torch.argmax(pred, dim=1)
			train_correct += torch.sum(torch.eq(pred, y)).item()

		# Validation
		with torch.no_grad():
			valid_correct = 0
			model.eval()
			for x, y in val_loader:
				x, y = x.to(args.device), y.to(args.device)
				pred = model(x, pred=True)
				pred = torch.argmax(pred, dim=1)
				valid_correct += torch.sum(torch.eq(pred, y)).item()

		logger.info('epoch:{}, Total train loss: {:.4f}, ' \
				'Train Acc: {:.4f},  Val. Acc: {:.4f}, ' \
				.format(epoch+1, train_loss, \
				train_correct / len_train_data,valid_correct / len_val_data))


		if best_accuracy < valid_correct:
			best_accuracy = valid_correct
			best_model = copy.deepcopy(model)

	# Freeze DoctorNet
	model = best_model
	model.annotators.requires_grad = False
	optimizer_weight = optim.Adam(model.weights.parameters(), lr=3e-2, weight_decay=0)

	logger.info('\n\nStart DoctorNet averaging weight training!')
	
	best_accuracy = 0
	for epoch in range(10):
		train_loss = 0.0
		train_correct = 0
		model.train()
		for x, y, annotation in train_loader:
			model.zero_grad()

			x, y, annotation = x.to(args.device), y.to(args.device), annotation.to(args.device)
			decisions, weights = model(x, weight=True)
			# Calculate loss of annotators' labeling
			mask = annotation == -1
			annotation_org = annotation
			# Calculate sum of one-hot encoded anotators' labels
			annotation = annotation + 1
			annotation = F.one_hot(annotation)
			annotation = annotation[:, :, 1:].float()
			kk = torch.sum(annotation, axis=1)
			ll = torch.sum((annotation_org != -1), axis=1)			
			annotation = torch.sum(annotation, axis=1) / ll[:,None]
			annotation = torch.argmax(annotation, dim=1)
			
			pred = torch.sum(decisions, axis=1)
			decisions = decisions.masked_fill(mask[:, :, None], 0)
			decisions = torch.sum(decisions, axis=1)

			weights = weights.masked_fill(mask, 0)
			weights = torch.sum(weights, axis=-1)
			decisions = decisions / weights[:, None]
			loss = criterion(decisions, annotation)

			loss.backward()
			optimizer_weight.step()
			train_loss += loss.item()

			pred = torch.argmax(pred, dim=1)
			train_correct += torch.sum(torch.eq(pred, y)).item()

		# Validation
		with torch.no_grad():
			valid_correct = 0
			model.eval()
			for x, y in val_loader:
				x, y = x.to(args.device), y.to(args.device)
				pred = model(x, pred=True, weight=True)
				pred = torch.argmax(pred, dim=1)
				valid_correct += torch.sum(torch.eq(pred, y)).item()

		logger.info('epoch:{}, Total train loss: {:.4f}, ' \
				'Train Acc: {:.4f},  Val. Acc: {:.4f}, ' \
				.format(epoch+1, train_loss, \
				train_correct / len_train_data,valid_correct / len_val_data))


		# Save the model with highest accuracy on validation set
		if best_accuracy < valid_correct:
			best_accuracy = valid_correct
			best_model = copy.deepcopy(model)

	
	out = {}
	out['best_val_acc']= best_accuracy
	out['final_model_dict']=best_model	
	return out
	
	
def test(args,alg_options,logger, best_model):

	test_loader = alg_options['test_loader']
	model_f=best_model
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
			f_x = model_f(batch_x.float(), pred=True, weight=True)
			y_hat = torch.max(f_x,1)[1]
			u = (y_hat == batch_y).sum()
			n_test_acc += u.item()
	logger.info('Final test accuracy : {:.4f}'.format(n_test_acc/len_test_data))
	return (n_test_acc/len_test_data)
