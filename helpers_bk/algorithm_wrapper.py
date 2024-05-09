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
from numpy.matlib import repmat
from torch.optim.lr_scheduler import MultiStepLR
# from algorithms.trainer_conal import *
# from algorithms.trainer_maxmig import *
# from algorithms.trainer_doctornet import *
from algorithms.trainer_crowdfeanet import *
# from algorithms.trainer_mbem import *
# from algorithms.trainer_crowdlayer import *
# from algorithms.trainer_dlmv import *
# from algorithms.trainer_dldsem import *
# from algorithms.trainer_traceregeecs import *
# from algorithms.trainer_noregeecs import *
from algorithms.trainer_geocrowdsnet import trainer_geocrowdsnet


def algorithmwrapperEECS(args,alg_options,logger):

    method            =alg_options['method']
    # Print args
    logger.info(args)

    if method=='TRACEREGEECS':
        test_acc = trainer_traceregeecs(args,alg_options,logger)
    elif method=='NOREGEECS':
        test_acc = trainer_noregeecs(args,alg_options,logger)
    elif method=='VOLMINEECS_LOGDETH_MV_INIT':
        test_acc = trainer_proposed_mv_init(args,alg_options,logger)
    elif method=='TRACEREGEECS_MV_INIT':
        test_acc = trainer_traceregeecs_mv_init(args,alg_options,logger)
    elif method=='CROWDLAYER':
        test_acc = trainer_crowdlayer(args,alg_options,logger)
    elif method=='MBEM':
        test_acc = trainer_mbem(args,alg_options,logger)
    elif method=='MBEM':
        test_acc = trainer_mbem(args,alg_options,logger)
    elif method=='DL_MAJORITYVOTING':
        test_acc = trainer_dlmv(args,alg_options,logger)
    elif method=='DL_DAWID_SKENE_EM':
        test_acc = trainer_dldsem(args,alg_options,logger)
    elif method=='GEOCROWDNET':
        test_acc = trainer_geocrowdsnet(args,alg_options,logger)
    elif method=='CROWDFEANET':
        test_acc = trainer_proposed(args,alg_options,logger)
    elif method=='CONAL':
        test_acc = trainer_conal(args,alg_options,logger)
    elif method =='MAXMIG':
        test_acc = trainer_maxmig(args,alg_options,logger)
    elif method =='DOCTOR_NET':
        train_data = LabelMeDatasetDoctorNet('data/LabelMe/train',is_train=True, annotator_index=annotators_sel)
        val_data = LabelMeDatasetDoctorNet('data/LabelMe/valid')
        test_data = LabelMeDatasetDoctorNet('data/LabelMe/test')
        alg_options['train_data']=train_data
        alg_options['val_data']=val_data
        alg_options['test_data']=test_data
        test_acc = trainer_doctornet(args,alg_options,logger)
    else:
        logger.info('Wrong method')
    return test_acc





