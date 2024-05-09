import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
from helpers.functions import multi_loss
import copy
from torch.utils.data import DataLoader
from helpers.model import *
import torch.optim as optim

def trainer_conal(args,alg_options,logger):

	
														  
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
		user_feature = np.eye(args.M)	
		model_f = CoNAL(args.device,'lenet',args.M, args.R, args.K, user_feature=user_feature, gumbel_common=False).cuda()		
		# The optimizers		
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)
		
	elif args.dataset=='fmnist':
		# Instantiate the model f and the model for confusion matrices		 
		user_feature = np.eye(args.M)	
		model_f = CoNAL(args.device,'resnet18f',args.M, args.R, args.K, user_feature=user_feature, gumbel_common=False).cuda()		
		# The optimizers		
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4)
	
	elif args.dataset=='labelme':
		# Instantiate the model f and the model for confusion matrices		 
		user_feature = np.eye(args.M)	
		model_f = CoNAL(args.device, args.M, args.R, args.K, user_feature=user_feature, gumbel_common=False).cuda()		
		# The optimizers		
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=0)

	elif args.dataset=='music':
		# Instantiate the model f and the model for confusion matrices		 
		user_feature = np.eye(args.M)	
		model_f = CoNAL_music(args.device,args.M, args.R, args.K, user_feature=user_feature, gumbel_common=False).cuda() 
		
		# The optimizers		
		optimizer_f = optim.Adam(model_f.parameters(), lr=args.learning_rate, weight_decay=0)
	elif args.dataset=='cifar10':
		# Instantiate the model f and the model for confusion matrices		 
		user_feature = np.eye(args.M)	
		model_f = CoNAL(args.device, args.M, args.R, args.K, user_feature=user_feature, gumbel_common=False).cuda()   
		# The optimizers
		optimizer_f = optim.SGD(model_f.parameters(), lr=args.learning_rate, weight_decay=1e-4,momentum=0.9)
		scheduler_f = MultiStepLR(optimizer_f, milestones=alg_options['milestones'], gamma=0.1)
	else:
		logger.info('Incorrect choice for dataset')
	if torch.cuda.is_available:
		model_f = model_f.to(device)
		
	# Loss function
	loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')


	


	best_valid_acc = 0
	best_model = None
	lr = 1e-2
	for epoch in range(args.n_epoch):
		train_acc = train(device, train_loader=train_loader, model=model_f, optimizer=optimizer_f, criterion=multi_loss, mode='common')
		valid_acc = test(args=args,alg_options=alg_options,logger=logger,best_model=model_f, flag_test=False)
		if valid_acc > best_valid_acc:
			best_valid_acc = valid_acc
			best_model = copy.deepcopy(model_f)
		logger.info('epoch:{},Train Acc: {:.4f},  Val. Acc: {:.4f}, ' \
				.format(epoch+1, train_acc,valid_acc))

	out={}
	out['final_model_f_dict']=best_model
	out['best_val_acc']= best_valid_acc
	return out


def train(device, train_loader, model, optimizer, criterion=F.cross_entropy, mode='simple', annotators=None, pretrain=None,
		  support = None, support_t = None, scale=0):
	loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
	model.train()
	correct = 0
	total = 0
	total_loss = 0
	loss = 0

	correct_rec = 0
	total_rec = 0
	for input, targets, targets_onehot, batch_annot_mask, batch_annot_list, true_labels in train_loader:  
		input = input.to(device)
		targets = targets.to(device).long()
		targets_onehot = targets_onehot.to(device)
		targets_onehot[targets_onehot == -1] = 0
		true_labels = true_labels.to(device).long()

		if mode == 'simple':
			loss = 0
			if scale:
				cls_out, output, trace_norm = model(input)
				loss += scale * trace_norm
				mask = targets != -1
				y_pred = torch.transpose(output, 1, 2)
				y_true = torch.transpose(targets_onehot, 1, 2).float()
				loss += torch.mean(-y_true[mask] * torch.log(y_pred[mask]))
			else:
				cls_out, output = model(input)
				loss += criterion(targets, output)
			_, predicted = cls_out.max(1)
			correct += predicted.eq(true_labels).sum().item()
			total += true_labels.size(0)
		elif mode == 'common':
			rec_loss = 0
			loss = 0
			cls_out, output = model(input.float(), mode='train')
			_, predicted = cls_out.max(1)
			correct += predicted.eq(true_labels).sum().item()
			total += true_labels.size(0)
			loss += criterion(targets, output)
			loss -= 0.00001 * torch.sum(torch.norm((model.kernel - model.common_kernel).view(targets.shape[1], -1), dim=1, p=2))
		else:
			output, _ = model(input.float())
			loss = loss_fn(output, true_labels)
			_, predicted = output.max(1)
			correct += predicted.eq(true_labels).sum().item()
			total += true_labels.size(0)
		total_loss += loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		return correct / total
		


def test(args,alg_options,logger,best_model,flag_test=True):
	if flag_test:
		test_loader = alg_options['test_loader']
	else:
		test_loader = alg_options['val_loader']
	model=best_model
	device=args.device

	
	
	with torch.no_grad():
		model.eval()
		correct = 0
		total = 0
		target = []
		predict = []
		for inputs, targets in test_loader:
			inputs = inputs.to(device)
			target.extend(targets.data.numpy())
			targets = targets.to(device)

			total += targets.size(0)
			output, _ = model(inputs.float(), mode='test')
			_, predicted = output.max(1)
			predict.extend(predicted.cpu().data.numpy())
			correct += predicted.eq(targets).sum().item()
		acc = correct / total
		f1 = f1_score(target, predict, average='macro')

	classes = list(set(target))
	classes.sort()
	acc_per_class = []
	predict = np.array(predict)
	target = np.array(target)
	for i in range(len(classes)):
		instance_class = target == i
		acc_i = np.mean(predict[instance_class] == classes[i])
		acc_per_class.append(acc_i)
	if flag_test:
		logger.info('Final test accuracy : {:.4f}'.format(acc))
	return acc




