import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
import IPython
from helpers.functions import multi_loss
import copy
from torch.autograd import Variable
from algorithms.common_maxmig import *
from torch.utils.data import DataLoader
from helpers.model import *
import torch.optim as optim

def trainer_maxmig(args,alg_options,logger):
						 
	learning_rate_list=alg_options['learning_rate_list']
	
	best_val_acc = 0
	# Perform training and validation
	logger.info('Training with learning_rate = 0.0001')
	out=train_val(args,alg_options,logger)
	if out['best_val_acc'] > best_val_acc:
		best_model = out['final_model_f_dict']
		best_val_acc=out['best_val_acc']	
				
	# Perform testing
	logger.info('Testing with learning_rate = 0.0001')
	test_acc = test_maxmig(args,alg_options,logger,best_model)
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
		p_pure = torch.FloatTensor(np.ones(args.K)/args.K)
		model_f = MAXMIG_left('lenet',args.R,args.K)
		model_A = MAXMIG_right(args.device,p_pure,alg_options['train_loader'],args.M,args.K)		
		
		# The optimizers		
		optimizer_f = optim.Adam(model_f.parameters(), lr=Config.left_learning_rate, weight_decay=1e-4)
		optimizer_A = optim.Adam(model_A.parameters(), lr=Config.right_learning_rate, weight_decay=0)
		
	elif args.dataset=='fmnist':
		# Instantiate the model f and the model for confusion matrices		 
		p_pure = torch.FloatTensor(np.ones(args.K)/args.K)
		model_f = MAXMIG_left('resnet18f',args.R,args.K)
		model_A = MAXMIG_right(args.device,p_pure,alg_options['train_loader'],args.M,args.K)		
		
		# The optimizers		
		optimizer_f = optim.Adam(model_f.parameters(), lr=Config.left_learning_rate, weight_decay=1e-4)
		optimizer_A = optim.Adam(model_A.parameters(), lr=Config.right_learning_rate, weight_decay=0)
		
	
	elif args.dataset=='labelme':
		# Instantiate the model f and the model for confusion matrices		 
		p_pure = torch.FloatTensor(np.ones(args.K)/args.K)
		model_f = MAXMIG_left('fcnn_dropout',args.R,args.K)
		model_A = MAXMIG_right(args.device,p_pure,alg_options['train_loader'],args.M,args.K)		
		
		# The optimizers		
		optimizer_f = optim.Adam(model_f.parameters(), lr=Config.left_learning_rate, weight_decay=0)
		optimizer_A = optim.Adam(model_A.parameters(), lr=Config.right_learning_rate, weight_decay=0)
		
	elif args.dataset=='music':
		# Instantiate the model f and the model for confusion matrices		 
		p_pure = torch.FloatTensor(np.ones(args.K)/args.K)
		model_f = MAXMIG_left_music()
		model_A = MAXMIG_right(args.device,p_pure,alg_options['train_loader'],args.M,args.K)		
		
		# The optimizers		
		optimizer_f = optim.Adam(model_f.parameters(), lr=Config.left_learning_rate, weight_decay=0)
		optimizer_A = optim.Adam(model_A.parameters(), lr=Config.right_learning_rate, weight_decay=0)

	elif args.dataset=='cifar10':
		# Instantiate the model f and the model for confusion matrices		 
		p_pure = torch.FloatTensor(np.ones(args.K)/args.K)
		model_f = MAXMIG_left('resnet9',args.R,args.K)
		model_A = MAXMIG_right(device,p_pure,alg_options['train_loader'],args.M,args.K)		
		
		# The optimizers		
		optimizer_f = optim.SGD(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4,momentum=0.9)
		optimizer_A = optim.SGD(model_A.parameters(), lr=args.learning_rate,weight_decay=0, momentum=0.9)
	else:
		logger.info('Incorrect choice for dataset')
	if torch.cuda.is_available:
		model_f = model_f.to(device)
		model_A = model_A.to(device)
		
	alg_options['model_f']=model_f
	alg_options['model_A']=model_A
	alg_options['optimizer_f']=optimizer_f
	alg_options['optimizer_A']=optimizer_A


	

	best_valid_acc = 0
	best_model = None
	p_pure = torch.FloatTensor(np.ones(args.K)/args.K)
	p = p_pure
	for epoch in range(args.n_epoch_maxmig):
		p_pure, p, train_acc = train_maxmig(epoch,p_pure,p,alg_options,args)
		valid_acc = test_maxmig(args=args,alg_options=alg_options,logger=logger,best_model=model_f, dataset_type='valid')
		if valid_acc > best_valid_acc:
			best_valid_acc = valid_acc
			best_model = copy.deepcopy(model_f)
		logger.info('epoch:{},Train Acc: {:.4f},  Val. Acc: {:.4f}, ' \
				.format(epoch+1, train_acc,valid_acc))

	out={}
	out['final_model_f_dict']=best_model
	out['best_val_acc']= best_valid_acc
	return out

def train_maxmig(epoch, priori_pure, priori,alg_options,args) :
	train_loader = alg_options['train_loader']
	left_model_pure=alg_options['model_f']
	right_model_pure=alg_options['model_A']
	left_optimizer_pure=alg_options['optimizer_f']
	right_optimizer_pure=alg_options['optimizer_A']
	left_model_pure.train()
	right_model_pure.train()

	pure_loss = 0
	co_loss1 = 0
	co_loss2 = 0

	p_pure = priori_pure
	p = priori

	for left_data, batch_annotations, batch_annot_onehot, batch_annot_mask, right_data, labels in train_loader:
		ep = Variable(right_data).float().to(args.device)

		images = Variable(left_data).float().to(args.device)
		# Pure Co-Training
		right_optimizer_pure.zero_grad()
		left_optimizer_pure.zero_grad()
		left_outputs = left_model_pure(images).cpu().float()
		right_outputs = right_model_pure(ep, left_outputs, prior=p_pure, type=2).cpu().float()
		#print("Left:",left_outputs,"Right:",right_outputs)
		loss = mig_loss_function(left_outputs, right_outputs, p_pure,args)
		loss.backward()
		right_optimizer_pure.step()
		left_optimizer_pure.step()
		pure_loss += loss.item()

	p_pure = torch.squeeze(right_model_pure.get_prior().detach().cpu())
	
	total_sample = 0   
	with torch.no_grad():
		left_model_pure.eval()
		right_model_pure.eval()
		total_corrects_pure = 0

		for images, batch_annotations, batch_annot_onehot, batch_annot_mask, ep, y_labels in train_loader:
			images = Variable(images).float().to(args.device)
			labels = y_labels.to(args.device)
			total_sample += images.size()[0]
			# Pure Cotraining
			outputs = left_model_pure(images)
			_, predicts = torch.max(outputs.data, 1)
			total_corrects_pure += torch.sum(predicts == labels)
	cotrain_acc = float(total_corrects_pure)/float(total_sample)


	return p_pure, p,cotrain_acc



def test_maxmig(args,alg_options,logger,best_model,dataset_type='test') :
	test_loader = alg_options['val_loader']
	model_f=alg_options['model_f']
	model_A=alg_options['model_A']
	if dataset_type=='test':
		model_f=best_model
		test_loader = alg_options['test_loader']
		
		
	total_sample = 0   
	with torch.no_grad():
		model_f.eval()
		model_A.eval()
		total_corrects_pure = 0

		for images,ep in test_loader:
			images = Variable(images).float().to(args.device)
			labels = ep.to(args.device)
			total_sample += images.size()[0]
			# Pure Cotraining
			outputs = model_f(images)
			_, predicts = torch.max(outputs.data, 1)
			total_corrects_pure += torch.sum(predicts == labels)
	cotrain_acc = float(total_corrects_pure)/float(total_sample)
	if dataset_type=='test':
		logger.info('Final test accuracy : {:.4f}'.format(cotrain_acc))
	return cotrain_acc
