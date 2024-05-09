import numpy as np
import torch.utils.data as Data
from PIL import Image
from helpers.transformer import *
import helpers.tools, pdb
from sklearn.metrics import confusion_matrix
from helpers.functions import *
from pathlib import Path
from numpy import genfromtxt
#def mnist_dataset(train=True, transform=None, target_transform=None, split_per=0.9, random_seed=1, num_class=10):
from numpy.matlib import repmat
from numpy.random import default_rng
from scipy import stats
from sklearn.model_selection import train_test_split

class cifar10_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, split_per=0.9 ,args=None,logger=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        original_images = np.load('data/cifar10/train_images.npy')
        original_labels = np.load('data/cifar10/train_labels.npy')
        num_class=10


        print(original_images.shape)
        logger.info('Splitting train and validation data')
        self.train_data, self.val_data, self.train_labels, self.val_labels,train_set_index = helpers.tools.dataset_split(original_images,original_labels, split_per, args.seed, args.K)




        length_train_data = len(self.train_data)
        length_valid_data = len(self.val_data)

        if self.train:
            if args.annotator_type=='synthetic':
                logger.info('Generating synthetic classifier annotations')
                self.A_true = generate_confusion_matrices(args,feature_size=(3,32,32))
                logger.info('Getting noisy labels from annotators')
                self.annotations_one_hot, self.annotations, self.flag_instance_indep_noise, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask, self.samples_noisy_label_count,self.samples_total_label_count = generate_annotator_labels(self.A_true,self.train_labels,self.train_data,(3,32,32),transform, args)
            elif args.annotator_type=='real':
                logger.info('Loading real annotations')
                annotations_all = np.load('data/cifar10n/annotations_cifar10n.npy')
                noise_label = torch.load('data/cifar10n/CIFAR-10_human.pt')
                random_label1 = noise_label['random_label1']
                random_label2 = noise_label['random_label2']
                random_label3 = noise_label['random_label3']
                annotations = np.zeros((annotations_all.shape[0],3))
                annotations[:,0] = random_label1
                annotations[:,1] = random_label2
                annotations[:,2] = random_label3
                print(annotations.shape)
                print(np.max(annotations))
                print(np.max(self.train_labels))
                annotations = annotations.astype(int)
                annotations=annotations[train_set_index,:]
                self.annotations_one_hot, self.annotations, self.flag_instance_indep_noise, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask, self.samples_noisy_label_count,self.samples_total_label_count = get_real_annotator_labels(annotations,args.K)
                self.A_true = estimate_confusion_matrices_from_groundtruth(self.annotations,self.train_labels)
            else:
                logger.info('Wrong choice')
            self.annotations_one_hot[self.annotations_one_hot==0] = args.coeff_label_smoothing/(args.K-1)
            self.annotations_one_hot[self.annotations_one_hot==1] = 1-args.coeff_label_smoothing

            self.train_data = self.train_data.reshape((length_train_data,3,32,32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
            print(self.train_data.shape)

        else:
            self.val_data = self.val_data.reshape((length_valid_data,3,32,32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        if self.train:
            img, annot, annot_one_hot, annot_mask, annot_list, label, flag_instance_indep, samples_noisy_label_count,samples_total_label_count \
                    = self.train_data[index], self.annotations[index], self.annotations_one_hot[index], self.annotator_mask[index], self.annotations_list_maxmig[index], \
                    self.train_labels[index], self.flag_instance_indep_noise[index],self.samples_noisy_label_count[index], self.samples_total_label_count[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]


        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return index, img, annot, annot_one_hot, annot_mask, annot_list, label,flag_instance_indep,samples_noisy_label_count,samples_total_label_count
        else:
            return img, label
    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class synthetic_data(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, split_per=0.9 ,args=None,logger=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        original_images = generate_data(args)
        original_labels = generate_f(args)
        num_class=10


        if self.train:
            if args.annotator_type=='synthetic':
                logger.info('Generating synthetic classifier annotations')
                self.A_true = generate_confusion_matrices(args,feature_size=(3,32,32))
                logger.info('Getting noisy labels from annotators')
                self.annotations_one_hot, self.annotations, self.flag_instance_indep_noise, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask = generate_annotator_labels(self.A_true,self.train_labels,self.train_data,(3,32,32),transform, args)
            elif args.annotator_type=='real':
                logger.info('Loading real annotations')
                annotations_all = np.load('data/cifar10n/annotations_cifar10n.npy')
                noise_label = torch.load('data/cifar10n/CIFAR-10_human.pt')
                random_label1 = noise_label['random_label1']
                random_label2 = noise_label['random_label2']
                random_label3 = noise_label['random_label3']
                annotations = np.zeros((annotations_all.shape[0],3))
                annotations[:,0] = random_label1
                annotations[:,1] = random_label2
                annotations[:,2] = random_label3
                print(annotations.shape)
                print(np.max(annotations))
                print(np.max(self.train_labels))
                annotations = annotations.astype(int)
                annotations=annotations[train_set_index,:]
                self.annotations_one_hot, self.annotations, self.flag_instance_indep_noise, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask = get_real_annotator_labels(annotations,args.K)
                self.A_true = estimate_confusion_matrices_from_groundtruth(self.annotations,self.train_labels)
            else:
                logger.info('Wrong choice')
            self.annotations_one_hot[self.annotations_one_hot==0] = args.coeff_label_smoothing/(args.K-1)
            self.annotations_one_hot[self.annotations_one_hot==1] = 1-args.coeff_label_smoothing

            self.train_data = self.train_data.reshape((length_train_data,3,32,32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
            print(self.train_data.shape)

        else:
            self.val_data = self.val_data.reshape((length_valid_data,3,32,32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        if self.train:
            img, annot, annot_one_hot, annot_mask, annot_list, label, flag_instance_indep = self.train_data[index], self.annotations[index], self.annotations_one_hot[index], self.annotator_mask[index], self.annotations_list_maxmig[index], self.train_labels[index], self.flag_instance_indep_noise[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]


        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return index, img, annot, annot_one_hot, annot_mask, annot_list, label,flag_instance_indep
        else:
            return img, label
    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)

class cifar10_test_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.test_data = np.load('data/cifar10/test_images.npy')
        self.test_labels = np.load('data/cifar10/test_labels.npy')
        self.test_data = self.test_data.reshape((10000,3,32,32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)


class cifar100_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, split_per=0.9 ,args=None,logger=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        original_images = np.load('data/cifar100/train_images.npy')
        original_labels = np.load('data/cifar100/train_labels.npy')
        num_class=10


        print(original_images.shape)
        logger.info('Splitting train and validation data')
        self.train_data, self.val_data, self.train_labels, self.val_labels,train_set_index = helpers.tools.dataset_split(original_images,original_labels, split_per, args.seed, args.K)




        length_train_data = len(self.train_data)
        length_valid_data = len(self.val_data)

        if self.train:
            if args.annotator_type=='synthetic':
                logger.info('Generating synthetic classifier annotations')
                self.A_true = generate_confusion_matrices(args,feature_size=(3,32,32))
                logger.info('Getting noisy labels from annotators')
                self.annotations_one_hot, self.annotations, self.flag_instance_indep_noise, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask, self.samples_noisy_label_count,self.samples_total_label_count = generate_annotator_labels(self.A_true,self.train_labels,self.train_data,(3,32,32),transform, args)
            elif args.annotator_type=='real':
                logger.info('Loading real annotations')
                annotations_all = np.load('data/cifar100n/annotations_cifar100n.npy')
                noise_label = torch.load('data/cifar100n/CIFAR-100_human.pt')
                random_label1 = noise_label['random_label1']
                random_label2 = noise_label['random_label2']
                random_label3 = noise_label['random_label3']
                annotations = np.zeros((annotations_all.shape[0],3))
                annotations[:,0] = random_label1
                annotations[:,1] = random_label2
                annotations[:,2] = random_label3
                print(annotations.shape)
                print(np.max(annotations))
                print(np.max(self.train_labels))
                annotations = annotations.astype(int)
                annotations=annotations[train_set_index,:]
                self.annotations_one_hot, self.annotations, self.flag_instance_indep_noise, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask, self.samples_noisy_label_count,self.samples_total_label_count = get_real_annotator_labels(annotations,args.K)
                self.A_true = estimate_confusion_matrices_from_groundtruth(self.annotations,self.train_labels)
            else:
                logger.info('Wrong choice')
            self.annotations_one_hot[self.annotations_one_hot==0] = args.coeff_label_smoothing/(args.K-1)
            self.annotations_one_hot[self.annotations_one_hot==1] = 1-args.coeff_label_smoothing

            self.train_data = self.train_data.reshape((length_train_data,3,32,32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
            print(self.train_data.shape)

        else:
            self.val_data = self.val_data.reshape((length_valid_data,3,32,32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        if self.train:
            img, annot, annot_one_hot, annot_mask, annot_list, label, flag_instance_indep, samples_noisy_label_count,samples_total_label_count \
                    = self.train_data[index], self.annotations[index], self.annotations_one_hot[index], self.annotator_mask[index], self.annotations_list_maxmig[index], \
                    self.train_labels[index], self.flag_instance_indep_noise[index],self.samples_noisy_label_count[index], self.samples_total_label_count[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]


        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return index, img, annot, annot_one_hot, annot_mask, annot_list, label,flag_instance_indep,samples_noisy_label_count,samples_total_label_count
        else:
            return img, label
    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)




class cifar100_test_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.test_data = np.load('data/cifar100/test_images.npy')
        self.test_labels = np.load('data/cifar100/test_labels.npy')
        self.test_data = self.test_data.reshape((10000,3,32,32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)



class mnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, split_per=0.9, random_seed=1,length_data = 20000,args=None,logger=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train


        original_images = np.load('data/mnist/train_images.npy')
        original_labels = np.load('data/mnist/train_labels.npy')
        num_class=10

        length_original_data = len(original_images)
        print('shape of original images')
        print(original_images.shape)
        logger.info('Splitting train and validation data')
        self.train_data, self.val_data, self.train_labels, self.val_labels,train_set_index = helpers.tools.dataset_split(original_images,original_labels, split_per, random_seed, num_class)


        length_train_data = len(self.train_data)
        length_valid_data = len(self.val_data)

        # Get some 10K samples to train the machine classifiers
        #Data allocation
        #for training machines = 5000
        # for training + validation classifier  = 5000,10000,15000
        # validation 95%
        # Allocating some data for annotations
        data_train_annotators, train_val_data, y_train_annotators, train_val_labels = train_test_split(original_images, original_labels, train_size=10000/length_original_data)
        print('shape of images for training annotators')
        print(data_train_annotators.shape)


        # Getting training and validation data
        train_val_data_selected, _, train_val_labels_selected, _ = train_test_split(train_val_data, train_val_labels, train_size=(length_data)/(length_original_data-10000))


        logger.info('Splitting train and validation data')
        self.train_data, self.val_data, self.train_labels, self.val_labels,train_set_index = helpers.tools.dataset_split(train_val_data_selected,train_val_labels_selected, split_per, random_seed, num_class)
        print('shape of images for training')
        print(self.train_data.shape)
        print('shape of images for validation')
        print(self.val_data.shape)

        if self.train:
            if args.annotator_type=='synthetic':
                logger.info('Generating synthetic classifier annotations')
                self.A_true = generate_confusion_matrices(args.M,args.K,args.gamma,args.conf_mat_type)
                logger.info('Getting noisy labels from annotators')
                self.annotations_one_hot, self.annotations, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask = generate_annotator_labels(self.A_true,args.annotator_label_pattern,args.p,args.l,self.train_labels)

            elif args.annotator_type=='machine-classifier':
                logger.info('Getting machine classifier annotations')
                self.annotations_one_hot, self.annotations, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask\
                        = generate_machine_classifier_annotations(data_train_annotators,\
                        y_train_annotators,self.train_data,self.train_labels,args,logger,self.transform,self.target_transform)
                #annotations_1 = np.argmax(annotations_one_hot,axis=2)
                self.A_true=estimate_confusion_matrices_from_groundtruth(self.annotations,self.train_labels)
            else:
                logger.info('Wrong choice')
            self.annotations_one_hot[self.annotations_one_hot==0] = args.coeff_label_smoothing/(args.K-1)
            self.annotations_one_hot[self.annotations_one_hot==1] = 1-args.coeff_label_smoothing




    def __getitem__(self, index):

        if self.train:
            img, annot, annot_one_hot, annot_mask, annot_list, label = self.train_data[index], self.annotations[index], self.annotations_one_hot[index], self.annotator_mask[index], self.annotations_list_maxmig[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]


        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return img, annot, annot_one_hot, annot_mask, annot_list, label
        else:
            return img, label
    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class mnist_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/mnist/test_images.npy')
        self.test_labels = np.load('data/mnist/test_labels.npy') - 1 # 0-9
        print(self.test_data.shape)

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)


class fmnist_dataset1(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, split_per=0.9, random_seed=1,length_data = 20000,args=None,logger=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train


        original_images = np.load('data/fmnist/train_images.npy')
        original_labels = np.load('data/fmnist/train_labels.npy')
        num_class=10

        length_original_data = len(original_images)
        print('shape of original images')
        print(original_images.shape)
        logger.info('Splitting train and validation data')
        self.train_data, self.val_data, self.train_labels, self.val_labels,train_set_index = helpers.tools.dataset_split(original_images,original_labels, split_per, random_seed, num_class)


        length_train_data = len(self.train_data)
        length_valid_data = len(self.val_data)

        if args.annotator_type=='machine-classifier':
            # Get some 10K samples to train the machine classifiers
                #Data allocation
                #for training machines = 5000
                # for training + validation classifier  = 5000,10000,15000
                # validation 95%
                # Allocating some data for annotations
                data_train_annotators, train_val_data, y_train_annotators, train_val_labels = train_test_split(original_images, original_labels, train_size=10000/length_original_data)
                print('shape of images for training annotators')
                print(data_train_annotators.shape)


                # Getting training and validation data
                train_val_data_selected, _, train_val_labels_selected, _ = train_test_split(train_val_data, train_val_labels, train_size=(length_data)/(length_original_data-10000))


                logger.info('Splitting train and validation data')
                self.train_data, self.val_data, self.train_labels, self.val_labels,train_set_index = helpers.tools.dataset_split(train_val_data_selected,train_val_labels_selected, split_per, random_seed, num_class)
                print('shape of images for training')
                print(self.train_data.shape)
                print('shape of images for validation')
                print(self.val_data.shape)

        if self.train:
            if args.annotator_type=='synthetic':
                logger.info('Generating synthetic classifier annotations')
                self.A_true = generate_confusion_matrices(args.M,args.K,args.gamma,args.conf_mat_type)
                logger.info('Getting noisy labels from annotators')
                self.annotations_one_hot, self.annotations, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask = generate_annotator_labels(self.A_true,args.annotator_label_pattern,args.p,args.l,self.train_labels)

            elif args.annotator_type=='machine-classifier':
                logger.info('Getting machine classifier annotations')
                self.annotations_one_hot, self.annotations, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask\
                        = generate_machine_classifier_annotations(data_train_annotators,\
                        y_train_annotators,self.train_data,self.train_labels,args,logger,self.transform,self.target_transform)
                #annotations_1 = np.argmax(annotations_one_hot,axis=2)
                self.A_true=estimate_confusion_matrices_from_groundtruth(self.annotations,self.train_labels)
            else:
                logger.info('Wrong choice')
            self.annotations_one_hot[self.annotations_one_hot==0] = args.coeff_label_smoothing/(args.K-1)
            self.annotations_one_hot[self.annotations_one_hot==1] = 1-args.coeff_label_smoothing




    def __getitem__(self, index):

        if self.train:
            img, annot, annot_one_hot, annot_mask, annot_list, label = self.train_data[index], self.annotations[index], self.annotations_one_hot[index], self.annotator_mask[index], self.annotations_list_maxmig[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]


        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return img, annot, annot_one_hot, annot_mask, annot_list, label
        else:
            return img, label
    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class fmnist_test_dataset1(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/fmnist/test_images.npy')
        self.test_labels = np.load('data/fmnist/test_labels.npy') - 1 # 0-9
        print(self.test_data.shape)

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)


class fmnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, split_per=0.9 ,args=None,logger=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        original_images = np.load('data/fmnist/train_images.npy')
        original_labels = np.load('data/fmnist/train_labels.npy')
        num_class=10


        print(original_images.shape)
        logger.info('Splitting train and validation data')
        self.train_data, self.val_data, self.train_labels, self.val_labels,train_set_index = helpers.tools.dataset_split(original_images,original_labels, split_per, args.seed, args.K)




        length_train_data = len(self.train_data)
        length_valid_data = len(self.val_data)

        if self.train:
            if args.annotator_type=='synthetic':
                logger.info('Generating synthetic classifier annotations')
                self.A_true = generate_confusion_matrices(args,feature_size=(28,28))
                logger.info('Getting noisy labels from annotators')
                self.annotations_one_hot, self.annotations, self.flag_instance_indep_noise, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask, self.samples_noisy_label_count,self.samples_total_label_count = generate_annotator_labels(self.A_true,self.train_labels,self.train_data,(28,28),transform, args)
            else:
                logger.info('Wrong choice')
            self.annotations_one_hot[self.annotations_one_hot==0] = args.coeff_label_smoothing/(args.K-1)
            self.annotations_one_hot[self.annotations_one_hot==1] = 1-args.coeff_label_smoothing

            #self.train_data = self.train_data.reshape((length_train_data,3,32,32))
            #self.train_data = self.train_data.transpose((0, 2, 3, 1))
            print(self.train_data.shape)

        else:
            self.val_data = self.val_data#.reshape((length_valid_data,3,32,32))
            #self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        if self.train:
            img, annot, annot_one_hot, annot_mask, annot_list, label, flag_instance_indep, samples_noisy_label_count,samples_total_label_count \
                    = self.train_data[index], self.annotations[index], self.annotations_one_hot[index], self.annotator_mask[index], self.annotations_list_maxmig[index], \
                    self.train_labels[index], self.flag_instance_indep_noise[index],self.samples_noisy_label_count[index], self.samples_total_label_count[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]


        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return index, img, annot, annot_one_hot, annot_mask, annot_list, label,flag_instance_indep,samples_noisy_label_count,samples_total_label_count
        else:
            return img, label
    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)




class fmnist_test_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.test_data = np.load('data/fmnist/test_images.npy')
        self.test_labels = np.load('data/fmnist/test_labels.npy')
        #self.test_data = self.test_data.reshape((10000,3,32,32))
        #self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)


class labelme_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, split_per=0.9, random_seed=1,length_train_data = 20000,args=None,logger=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.train_data = np.load('data/LabelMe/data_train_vgg16.npy')
        self.train_labels = np.load('data/LabelMe/labels_train.npy')
        self.val_data = np.load('data/LabelMe/data_valid_vgg16.npy')
        self.val_labels= np.load('data/LabelMe/labels_valid.npy')




        length_train_data = len(self.train_data)
        length_valid_data = len(self.val_data)

        if self.train:
            if args.annotator_type=='real':
                logger.info('Loading real annotations')
                annotations = np.load('data/LabelMe/answers.npy')
                annotations = annotations.astype(int)
                self.annotations_one_hot, self.annotations, self.flag_instance_indep_noise, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask, self.samples_noisy_label_count,self.samples_total_label_count = get_real_annotator_labels(annotations,args.K)
                self.A_true = estimate_confusion_matrices_from_groundtruth(self.annotations,self.train_labels)
            else:
                logger.info('Wrong choice')
            self.annotations_one_hot[self.annotations_one_hot==0] = args.coeff_label_smoothing/(args.K-1)
            self.annotations_one_hot[self.annotations_one_hot==1] = 1-args.coeff_label_smoothing

            print(self.train_data.shape)



    def __getitem__(self, index):

        if self.train:
            img, annot, annot_one_hot, annot_mask, annot_list, label, flag_instance_indep, samples_noisy_label_count,samples_total_label_count \
                    = self.train_data[index], self.annotations[index], self.annotations_one_hot[index], self.annotator_mask[index], self.annotations_list_maxmig[index], \
                    self.train_labels[index], self.flag_instance_indep_noise[index],self.samples_noisy_label_count[index], self.samples_total_label_count[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return index, img, annot, annot_one_hot, annot_mask, annot_list, label,flag_instance_indep,samples_noisy_label_count,samples_total_label_count
        else:
            return img, label
    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)

class labelme_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/LabelMe/data_test_vgg16.npy')
        self.test_labels= np.load('data/LabelMe/labels_test.npy')

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)



class music_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, split_per=0.9, random_seed=1,length_train_data = 20000,args=None,logger=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train


        self.train_data_label = genfromtxt('data/Music/music_genre_gold.csv',delimiter=',')
        self.train_data_label1 = genfromtxt('data/Music/music_genre_gold.csv',dtype=str, delimiter=',')
        self.train_data = self.train_data_label[1:,1:-1]
        N = np.shape(self.train_data)[0]

        self.mean_train = np.mean(self.train_data,axis=0)
        self.std_train = np.std(self.train_data,axis=0)
        self.std_train[self.std_train == 0] = 1.0
        self.train_data = np.divide((self.train_data - repmat(self.mean_train,N,1)), repmat(self.std_train,N,1))


        self.train_data_id = self.train_data_label1[1:,0]
        self.train_label_str = self.train_data_label1[1:,-1]
        self.label_list = np.unique(self.train_label_str)
        self.train_labels = np.zeros(np.shape(self.train_label_str))
        for i in range(args.K):
            self.train_labels[self.train_label_str==self.label_list[i]]=i




        self.train_labels = self.train_labels.astype(int)
        self.train_data, self.val_data, self.train_labels, self.val_labels, train_set_index = helpers.tools.dataset_split(self.train_data,self.train_labels, split_per, random_seed, args.K)



        length_train_data = len(self.train_data)
        length_valid_data = len(self.val_data)

        if self.train:
            if args.annotator_type=='real':
                logger.info('Loading real annotations')
                self.annotations_data_all = genfromtxt('data/Music/music_genre_mturk.csv',dtype=str,delimiter=',')
                self.annotator_id_all = self.annotations_data_all[1:,1]
                self.train_data_id_annotated_all = self.annotations_data_all[1:,0]
                self.annotations_str = self.annotations_data_all[1:,-1]
                self.annotations_raw_number = np.zeros(np.shape(self.annotations_str))
                for i in range(args.K):
                    self.annotations_raw_number[self.annotations_str==self.label_list[i]]=i
                self.annotator_list = np.unique(self.annotator_id_all)
                M = np.shape(self.annotator_list)[0]
                self.annotations_all = -1*np.ones((N,M))
                for j in range(np.shape(self.annotations_raw_number)[0]):
                    n = np.argwhere(self.train_data_id ==self.train_data_id_annotated_all[j])
                    m =  np.argwhere(self.annotator_list ==self.annotator_id_all[j])
                    self.annotations_all[n,m] = self.annotations_raw_number[j]
                self.annotations_all = self.annotations_all.astype(int)
                self.annotations=self.annotations_all
                self.annotations=self.annotations[train_set_index,:]
                self.annotations_one_hot, self.annotations, self.annotations_list_maxmig, \
                        self.annotator_softmax_label_mbem,self.annotators_per_sample_mbem,self.annotator_mask = get_real_annotator_labels(self.annotations,args.K)
                self.A_true = estimate_confusion_matrices_from_groundtruth(self.annotations,self.train_labels)
            else:
                logger.info('Wrong choice')
            self.annotations_one_hot[self.annotations_one_hot==0] = args.coeff_label_smoothing/(args.K-1)
            self.annotations_one_hot[self.annotations_one_hot==1] = 1-args.coeff_label_smoothing

            print(self.train_data.shape)



    def __getitem__(self, index):

        if self.train:
            img, annot, annot_one_hot, annot_mask, annot_list, label = self.train_data[index], self.annotations[index], self.annotations_one_hot[index], self.annotator_mask[index], self.annotations_list_maxmig[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.train:
            return img, annot, annot_one_hot, annot_mask, annot_list, label
        else:
            return img, label
    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)

class music_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None, args=None, mean_train=None, std_train=None, label_list= None):

        self.transform = transform
        self.target_transform = target_transform
        self.test_data_label = genfromtxt('data/Music/music_genre_test.csv',delimiter=',')
        self.test_data_label1 = genfromtxt('data/Music/music_genre_test.csv',dtype=str,delimiter=',')
        self.test_data = self.test_data_label[1:,1:-1]
        N_test = np.shape(self.test_data)[0]
        self.test_data = np.divide((self.test_data - repmat(mean_train,N_test,1)), repmat(std_train,N_test,1))

        self.test_label_str = self.test_data_label1[1:,-1]
        self.test_labels = np.zeros(np.shape(self.test_label_str))
        for i in range(args.K):
            self.test_labels[self.test_label_str==label_list[i]]=i

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)




