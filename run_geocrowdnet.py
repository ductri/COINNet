import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from helpers.functions import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from helpers.data_load import *
from helpers.transformer import *
from helpers.algorithm_wrapper import algorithmwrapperEECS
import os
from datetime import datetime
from numpy.random import default_rng
import random
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import wandb
from hydra import compose, initialize
import hydra
from omegaconf import OmegaConf

from utils import create_ray_wrapper, create_wandb_wrapper
from my_dataset import CIFAR10ShahanaModule, LitCIFAR10MachineAnnotations, get_dataset

from ray.util import inspect_serializability

class FakeLogger:
    def info(self, s):
        print(s)


def my_main(conf, unique_name):
    print(OmegaConf.to_yaml(conf))

    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id',type=int,help='Session_ID',default=3)

    parser.add_argument('--M',type=int,help='No of annotators',default=3)
    parser.add_argument('--K',type=int,help='No of classes',default=10)
    parser.add_argument('--N',type=int,help='No of data samples (synthetic data)',default=50000)
    parser.add_argument('--R',type=int,help='Dimension of data samples (synthetic data)',default=5)
    parser.add_argument('--l',type=int,help='number of annotations per sample or number of samples per annotators',default=3)
    parser.add_argument('--flag_class_specialists',type=bool,help='True or False',default=False)
    parser.add_argument('--dataset',type=str,help='synthetic or cifar10 or mnist',default='cifar10')
    parser.add_argument('--annotator_type',type=str,help='synthetic, machine-classifier, good-bad-annotator-mix or real',default='synthetic')
    parser.add_argument('--instance_indep_conf_type',type=str,help='symmetric,separable_uniform',default='symmetric_average')
    parser.add_argument('--flag_preload_annotations',type=bool,help='True or False (if True, load annotations from file, otherwise generate annotations',\
            default=True)
    parser.add_argument('--varepsilon',type=float,help='parameter for class specialists',default=0.001)
    parser.add_argument('--lam',type=float,help='Volume regularizer',default=0)
    parser.add_argument('--lam_trace',type=float,help='Volume regularizer',default=0.01)
    parser.add_argument('--mu',type=float,help='instance dependent regularizer',default=0)
    parser.add_argument('--seed',type=int,help='Random seed',default=np.random.randint(100000))
    parser.add_argument('--device',type=int,help='GPU device number',default=0)
    parser.add_argument('--n_trials',type=int,help='No of trials',default=1)
    parser.add_argument('--flag_hyperparameter_tuning',type=bool,help='True or False',default=False)
    parser.add_argument('--proposed_init_type',type=str,help='close_to_identity or mle_based or identity',default='identity')
    parser.add_argument('--proposed_projection_type',type=str,help='simplex_projection or softmax or sigmoid_projection',default='simplex_projection')
    parser.add_argument('--classifier_NN',type=str,help='resnet9 or resnet18 or resnet34',default='resnet9')
    parser.add_argument('--weight_decay', type=float, help='weight_decay for the optimizer', default=1e-3)

    parser.add_argument('--total_noise_rate', type = float, help = 'overall noise rate for the labels', default =0.2)
    parser.add_argument('--percent_instance_noise', type = float, help = 'percent of samples having instance-dependent noise', default =0.3) # TRI, debug
    parser.add_argument('--vol_reg_type',type=str, default='max_logdeth')
    parser.add_argument('--p',type=float, default=0.2,help = 'paramater for p norm')
    parser.add_argument('--confusion_network_input_type', type=str, help = 'classifier_ouput or feature_embedding', default ='feature_embedding')
    parser.add_argument('--warmup_epoch',type=int,help='Number of Epochs for warmup',default=10)
    parser.add_argument('--flag_warmup', type = int, default =0)
    parser.add_argument('--flag_instance_dep_modeling', type = int, default =1)
    parser.add_argument('--flag_two_optimizers', type = int, default =0)
    parser.add_argument('--flag_instance_dep_score_calc', type = int, default =0) # Tri, debug
    parser.add_argument('--flag_underparameterized_instance_dep_modeling', type = int, default =0)
    parser.add_argument('--instance_dep_percent_estim', type = float, default =0.3)

    parser.add_argument('--learning_rate',type=float,help='Learning rate',default=0.001)
    parser.add_argument('--batch_size',type=int,help='Batch Size',default=100)
    parser.add_argument('--n_epoch',type=int,help='Number of Epochs',default=200)
    parser.add_argument('--coeff_label_smoothing',type=float,help='label smoothing coefficient',default=0)
    parser.add_argument('--log_folder',type=str,help='log folder path',default='tmp/cifar10_synthetic/')

    parser.add_argument('--flag_wandb',type = int, default =1)

    # Parser
    args=parser.parse_args([])

    # project_name = 'shahana_outlier'
    # with wandb.init(entity='narutox', project=project_name, tags=conf.tags, config=OmegaConf.to_container(conf, resolve=True)) as run:
    args.dataset = conf.data.dataset

    args.batch_size = conf.train.batch_size
    args.n_epoch = conf.train.num_epochs
    args.lam = conf.train.lam
    args.K = conf.data.K
    args.N = conf.data.N
    args.M = conf.data.M
    args.l = conf.data.M
    args.fnet_type = conf.fnet_type
    args.classifier_NN = conf.fnet_type
    args.vol_reg_type = conf.train.vol_reg_type
    args.learning_rate = conf.train.lr

    # Setting GPU and cuda settings
    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available:
        device = torch.device('cuda:'+str(args.device))
    torch.autograd.set_detect_anomaly(True)

    # Log file settings
    time_now = datetime.now()
    time_now.strftime("%b-%d-%Y")
    log_file_name = args.log_folder+'log_'+str(time_now.strftime("%b-%d-%Y"))+'_synthetic_cifar10_'+str(args.session_id)+'.txt'
    result_file=args.log_folder+'result_'+str(time_now.strftime("%b-%d-%Y"))+'_synthetic_cifar10_'+str(args.session_id)+'.txt'

    # if os.path.exists(log_file_name):
    #     os.remove(log_file_name)
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    # # fh = logging.FileHandler(log_file_name)
    # ch = logging.StreamHandler()
    # formatter = logging.Formatter('%(message)s')
    # # fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # # logger.addHandler(fh)
    # logger.addHandler(ch)
    logger = FakeLogger()


    def main():
        rng = default_rng()

        #Set algorithm flags
        # algorithms_list = ['CROWDLAYER']
        algorithms_list = ['GEOCROWDNET']

        # Data logging variables
        test_acc_all = np.ones((len(algorithms_list),args.n_trials))*np.nan
        # fileid = open(result_file,"w")
        # fileid.write('#########################################################\n')
        # fileid.write(str(time_now))
        # fileid.write('\n')
        # fileid.write('Trial#\t')
        # for s in algorithms_list:
        #     fileid.write(s+str('\t'))
        # fileid.write('\n')

        annotators_sel = range(args.M)
        alg_options = {
                'device':device,
                'loss_function_type':'cross_entropy'}

        #args.n_epoch=80

        alg_options['lamda_list']=[args.lam]
        alg_options['mu_list']=[args.mu]
        alg_options['learning_rate_list']=[args.learning_rate]

        if args.dataset=='cifar10':
            alg_options['flag_lr_scheduler'] = True
            alg_options['milestones'] = [30,60]
        else:
            alg_options['flag_lr_scheduler'] = False
            alg_options['milestones'] = [30,60]

        for t in range(args.n_trials):
            np.random.seed(t+args.seed)
            torch.manual_seed(t+args.seed)
            torch.cuda.manual_seed(t+args.seed)
            random.seed(t+args.seed)

            # # Get the train, validation and test dataset
            # train_data     = Cifar10Dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,split_per=0.95,args=args,logger=logger)
            # val_data     = Cifar10Dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,split_per=0.95,args=args,logger=logger)
            # test_data     = cifar10_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)

            # alg_options['train_data']=train_data
            # alg_options['A_true']=train_data.A_true
            # alg_options['data_train']=train_data.train_data
            # alg_options['y_train']=train_data.train_labels
            # alg_options['data_val']=val_data.val_data
            # alg_options['y_val']=val_data.val_labels
            # alg_options['data_test']=test_data.test_data
            # alg_options['y_test']=test_data.test_labels
            # alg_options['annotations']=train_data.annotations
            # alg_options['annotations_one_hot']=train_data.annotations_one_hot
            # alg_options['annotator_softmax_label_mbem']=train_data.annotator_softmax_label_mbem
            # alg_options['annotators_per_sample_mbem']=train_data.annotators_per_sample_mbem
            # alg_options['annotations_list_maxmig']=train_data.annotations_list_maxmig
            # alg_options['annotators_sel']=annotators_sel
            # alg_options['annotator_mask']=train_data.annotator_mask

            #args.M = train_data.annotations.shape[1]
            # args.N = train_data.train_labels.shape[0]
            # args.K = 10

            # Prepare data for training/validation and testing
            # train_loader = DataLoader(dataset=train_data,
            #         batch_size=args.batch_size,
            #         num_workers=3,
            #         shuffle=True,
            #         drop_last=True,
            #         pin_memory=True)
            # val_loader = DataLoader(dataset=val_data,
            #         batch_size=args.batch_size,
            #         num_workers=3,
            #         shuffle=False,
            #         drop_last=True,
            #         pin_memory=True)
            # test_loader = DataLoader(dataset=test_data,
            #         batch_size=args.batch_size,
            #         num_workers=3,
            #         shuffle=False,
            #         drop_last=True,
            #         pin_memory=True)

            #
            # if conf.data.dataset == 'cifar10_machine':
            #     data_module = LitCIFAR10MachineAnnotations(conf.train.batch_size, root=conf.root, filename=conf.data.filename)
            # elif conf.data.dataset == 'cifar10':
            #     data_module = CIFAR10ShahanaModule(conf)
            #     args.percent_instance_noise = conf.data.percent_instance_noise
            #     args.instance_indep_conf_type = conf.data.instance_indep_conf_type
            #     args.total_noise_rate = conf.data.total_noise_rate
            # else:
            #     raise Exception(f'dataset typo: {conf.data.dataset}')
            data_module = get_dataset(conf)
            data_module.prepare_data()
            data_module.setup('fit')
            data_module.setup('test')
            alg_options['train_loader'] = data_module.train_dataloader()
            alg_options['val_loader'] = data_module.val_dataloader()
            alg_options['test_loader']= data_module.test_dataloader()
            alg_options['conf']= conf

            #################################Run Algorithms#######################################
            logger.info('Starting trial '+str(t)+'.....................')
            for k in range(len(algorithms_list)):
                logger.info('Running '+algorithms_list[k])
                alg_options['method']=algorithms_list[k]
                test_acc=algorithmwrapperEECS(args,alg_options,logger)
                test_acc_all[k,t]=test_acc*100
                wandb.summary['test_acc'] = test_acc

    main()



@hydra.main(version_base=None, config_path=f"conf", config_name="config_geocrowdnet")
def main_hydra(conf):
    print(OmegaConf.to_yaml(conf))

    runner = create_wandb_wrapper(my_main)
    if conf.with_ray:
        runner = create_ray_wrapper(runner)
    runner(conf)


if __name__ == '__main__':
    main_hydra()

    # x= my_main
    # inspect_serializability(x, name="test")
    # print('xx')

