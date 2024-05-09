from tkinter import YES
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.hub as hub
import torchvision.models as models
from torch.autograd import Variable
import math
from helpers.vgg import *
from torchvision import transforms
from algorithms.functional_conal import *
from algorithms.common_maxmig import *


class FCNN(nn.Module):
    def __init__(self,R,K,hidden_units,hidden_layers):
        super(FCNN,self).__init__()
        layer_list=[]
        n_in=R
        for i in range(hidden_layers):
            layer_list.append(nn.Linear(n_in,hidden_units))
            layer_list.append(nn.ReLU(inplace=False))
            n_in = hidden_units
        layer_list.append(nn.Linear(hidden_units,K))
        self.layers=nn.Sequential(*layer_list)

    def forward(self,x):
        x = self.layers(x)
        p = F.softmax(x,dim=1)
        return p



#class confusion_matrices(nn.Module):
#    def __init__(self, device, M, K, init=-0.48):
#        super(confusion_matrices, self).__init__()
#        self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(M, K, K)))
#        self.w.to(device)
#        co = torch.ones(K, K)
#        ind = np.diag_indices(co.shape[0])
#        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
#        self.co = co.to(device)
#        self.identity = torch.eye(K).to(device)
#        self.M = M
#        self.K = K
#
#
#    def forward(self):
#        A = torch.zeros(self.M,self.K,self.K)
#        for i in range(self.M):
#            sig = torch.sigmoid(self.w[i])-0.5
#            A[i] = self.identity.detach() + sig*self.co.detach()
#            A[i] = F.normalize(A[i].clone(), p=1, dim=0)
#        return A

class confusion_matrix_layer(nn.Module):
    def __init__(self,M,K):
        super(confusion_matrix_layer,self).__init__()
        #self.A = nn.ModuleList([nn.Linear(K,K,bias=False) for i in range(M)])
        self.A = nn.Parameter(torch.stack([torch.eye(K, K) for _ in range(M)]), requires_grad=True)
        self.M=M
        self.K = K
    def forward(self,x):
        y = torch.einsum('ik, bkj -> ibj',x,self.A)
        y = F.softmax(y,dim=2)
        return(y)

class CrowdLayer(nn.Module):
    def __init__(self,input_dim,M,K,fnet_type):
        super(CrowdLayer,self).__init__()
        if fnet_type=='fcnn_dropout':
            self.fnet = FCNN_Dropout(input_dim,K)
        elif fnet_type=='lenet':
            self.fnet = Lenet()
        elif fnet_type=='fcnn_dropout_batchnorm':
            self.fnet = FCNN_Dropout_BatchNorm(input_dim,K)
        elif fnet_type=='linear':
            self.fnet = LinearClassifier(input_dim,K)
        elif fnet_type=='resnet9':
            self.fnet = ResNet9(K)
        elif fnet_type=='resnet18':
            self.fnet = ResNet18(K)
        elif fnet_type=='resnet18f':
            self.fnet = ResNet18_F(K)
        elif fnet_type=='resnet34':
            self.fnet = ResNet34(K)
        else:
            self.fnet = FCNN_Dropout()
        self.A = nn.Parameter(torch.stack([torch.eye(K, K) for _ in range(M)]), requires_grad=True)
    def forward(self,x):
        x,g = self.fnet(x)
        y = torch.einsum('ij, bkj -> ibk',x,self.A)
        y = F.softmax(y,dim=2)
        return(x,y)


class CrowdNetwork(nn.Module):
    def __init__(self,args,A_init):
        input_dim=args.R
        fnet_type=args.classifier_NN
        init_method=args.proposed_init_type
        M=args.M
        K=args.K
        super(CrowdNetwork,self).__init__()
        if fnet_type=='fcnn_dropout':
            self.fnet = FCNN_Dropout(input_dim,K)
        elif fnet_type=='lenet':
            self.fnet = Lenet()
        elif fnet_type=='fcnn_dropout_batchnorm':
            self.fnet = FCNN_Dropout_BatchNorm(input_dim,K)
        elif fnet_type=='linear':
            self.fnet = LinearClassifier(input_dim,K)
        elif fnet_type=='resnet9':
            self.fnet = ResNet9(K)
        elif fnet_type=='resnet18f':
            self.fnet = ResNet18_F(K)
        elif fnet_type=='resnet18':
            self.fnet = ResNet18(K)
        elif fnet_type=='resnet34':
            self.fnet = ResNet34(K)
        else:
            self.fnet = FCNN_Dropout()
        if init_method=='close_to_identity':
            self.P = nn.Parameter(torch.stack([6*torch.eye(K)-5 for _ in range(M)]), requires_grad=True)
        elif init_method=='mle_based':
            self.P = nn.Parameter(torch.stack([A_init[m] for m in range(M)]), requires_grad=True)
        else:
            print('##################')
            self.P = nn.Parameter(torch.stack([torch.eye(K) for _ in range(M)]), requires_grad=True)
        #self.A = None
    def forward(self,x):
        x,g = self.fnet(x)
        #A = F.softplus(self.P)
        #A = F.normalize(A,p=1,dim=1)
        A = F.softmax(self.P,dim=1)
        y = torch.einsum('ij, bkj -> ibk',x,A)
        #y = F.softmax(y,dim=2)
        #E=torch.tensor(A)
        return(x,y,A)


class InstanceDependentCrowdNetwork1(nn.Module):
    def __init__(self,input_dim,M,K,fnet_type,init_method,A_init,flag_instance_dep_modeling,confusion_network_input_type):
        super(InstanceDependentCrowdNetwork1,self).__init__()
        if fnet_type=='fcnn_dropout':
            self.fnet = FCNN_Dropout(input_dim,K)
        elif fnet_type=='lenet':
            self.fnet = Lenet()
        elif fnet_type=='fcnn_dropout_batchnorm':
            self.fnet = FCNN_Dropout_BatchNorm(input_dim,K)
        elif fnet_type=='linear':
            self.fnet = LinearClassifier(input_dim,K)
        elif fnet_type=='resnet9':
            self.fnet = ResNet9(K)
        elif fnet_type=='resnet18f':
            self.fnet = ResNet18_F(K)
        elif fnet_type=='resnet18':
            self.fnet = ResNet18(K)
        elif fnet_type=='resnet34':
            self.fnet = ResNet34(K)
        else:
            self.fnet = FCNN_Dropout()
        if init_method=='close_to_identity':
            self.P = nn.Parameter(torch.stack([6*torch.eye(K)-5 for _ in range(M)]), requires_grad=True)
        elif init_method=='mle_based':
            self.P = nn.Parameter(torch.stack([A_init[m] for m in range(M)]), requires_grad=True)
        else:
            print(init_method)
            self.P = nn.Parameter(torch.stack([torch.eye(K) for _ in range(M)]), requires_grad=True)


        self.fc = nn.ModuleList([nn.Linear(K, K * K, bias=False) for i in range(M)])
        #self.fc = nn.Linear(K, K * K)
        self.M=M
        self.K=K
        self.flag_instance_dep_modeling=flag_instance_dep_modeling
        self.confusion_network_input_type=confusion_network_input_type
        self.w = torch.ones(K*K,K)*0
        for i in range(M):
            self.fc[i].weight.data= self.w
    def forward(self,x):
        x,y = self.fnet(x)
        #A = F.softplus(self.P)
        #A = F.normalize(A,p=1,dim=1)
        A = F.softmax(self.P,dim=1)
        if self.flag_instance_dep_modeling==1:
            if self.confusion_network_input_type == 'classifier_ouput':
                inp = x
            else:
                inp = y
            E = torch.stack([self.fc[i](inp) for i in range(self.M)])
            #E=torch.clip(E, min=1e-12)
            #E = torch.stack([self.fc(x) for i in range(self.M)])
            E = E.view(self.M, x.size(0), self.K, -1)
            #E = F.normalize(E, p=2, dim=2)
            A_mod= A.view(A.size(0),1,A.size(1),A.size(2))
            T_mod = torch.tensor(E+A_mod)
            #T_mod = F.softplus(T_mod)
            T_mod=torch.clip(T_mod, min=1e-12)
            T = F.normalize(T_mod,p=1,dim=2)
            E_mod = T-A_mod
            y = torch.einsum('ij, bikj -> ibk',x,T)
#             print("#################################################")
#             print(A_mod[0,0,2,:])
#             print(E[0,0,2,:])
#             print(T_mod[0,0,2,:])
#             print(A_mod[0,0,2,:])
#             print(E[0,3,2,:])
#             print(T_mod[0,3,2,:])
#             print("#################################################")
        else:
            E_mod=A
            y = torch.einsum('ij, bkj -> ibk',x,A)
        #y = torch.einsum('bijk, ik -> ibk',T,y)
        #E_mod=torch.tensor(A)
        #y = torch.einsum('ij, bkj -> ibk',x,A)
        #y = F.softmax(y,dim=2)
        #print(E_mod[0,4])
        return(x,y,A,E_mod)


class InstanceDependentCrowdNetwork(nn.Module):
    def __init__(self,args,A_init):
        super(InstanceDependentCrowdNetwork,self).__init__()
        input_dim=args.R
        fnet_type=args.classifier_NN
        init_method=args.proposed_init_type
        self.M=args.M
        self.K=args.K
        self.flag_instance_dep_modeling=args.flag_instance_dep_modeling
        self.confusion_network_input_type=args.confusion_network_input_type
        self.flag_underparameterized_instance_dep_modeling=args.flag_underparameterized_instance_dep_modeling
        M=args.M
        K=args.K
        if fnet_type=='fcnn_dropout':
            self.fnet = FCNN_Dropout(input_dim,K)
        elif fnet_type=='lenet':
            self.fnet = Lenet()
        elif fnet_type=='fcnn_dropout_batchnorm':
            self.fnet = FCNN_Dropout_BatchNorm(input_dim,K)
        elif fnet_type=='linear':
            self.fnet = LinearClassifier(input_dim,K)
        elif fnet_type=='resnet9':
            self.fnet = ResNet9(K)
        elif fnet_type=='resnet18f':
            self.fnet = ResNet18_F(K)
        elif fnet_type=='resnet18':
            self.fnet = ResNet18(K)
        elif fnet_type=='resnet34':
            self.fnet = ResNet34(K)
        else:
            self.fnet = FCNN_Dropout()
        if init_method=='close_to_identity':
            self.P = nn.Parameter(torch.stack([6*torch.eye(K)-5 for _ in range(M)]), requires_grad=True)
        elif init_method=='mle_based':
            self.P = nn.Parameter(torch.stack([A_init[m] for m in range(M)]), requires_grad=True)
        else:
            print(init_method)
            self.P = nn.Parameter(torch.stack([torch.eye(K) for _ in range(M)]), requires_grad=True)


        if self.flag_underparameterized_instance_dep_modeling:
            self.fc = nn.ModuleList([nn.Linear(K, K,bias=False) for i in range(M)])
            self.w = torch.ones(K,K)*0
        else:
            self.fc = nn.ModuleList([nn.Linear(K, K*K,bias=False) for i in range(M)])
            self.w = torch.ones(K*K,K)*0
        #self.fc = nn.Linear(K, K * K)


        for i in range(M):
            self.fc[i].weight.data= self.w
            #self.fc[i].bias.data= self.w
    def forward(self,x):
        x,y = self.fnet(x)
        A = F.softplus(self.P)
        A = F.normalize(A,p=1,dim=1)
        #A = F.softmax(self.P,dim=1)
        if self.flag_instance_dep_modeling==1:
            if self.confusion_network_input_type == 'classifier_ouput':
                inp = x
            else:
                inp = y
            if self.flag_underparameterized_instance_dep_modeling:
                print("I am here")
                Ef_x = torch.stack([self.fc[i](inp) for i in range(self.M)])
                Ef_x=torch.clip(Ef_x, min=-1,max=1)
                Ef_x = Ef_x.view(x.size(0),self.M, self.K)
                Af_x = torch.einsum('ij, bkj -> ibk',x,A)
                Tf_x = Af_x+Ef_x
                #Tf_x = F.softplus(Tf_x)
                Tf_x=torch.clip(Tf_x, min=1e-12)
                y = F.normalize(Tf_x,p=1,dim=2)
                E_mod = y-Af_x
            else:
                E = torch.stack([self.fc[i](inp) for i in range(self.M)])
                E=torch.clip(E, min=-1,max=1)
                #E = torch.stack([self.fc(x) for i in range(self.M)])
                E = E.view(self.M, x.size(0), self.K, -1)
                #E = F.normalize(E, p=2, dim=2)
                A_mod= A.view(A.size(0),1,A.size(1),A.size(2))
                T_mod = torch.tensor(E+A_mod)
                #T_mod = F.softplus(T_mod)
                T_mod=torch.clip(T_mod, min=1e-12)
                T = F.normalize(T_mod,p=1,dim=2)
                y = torch.einsum('ij, bikj -> ibk',x,T)
                E_mod = T-A_mod
                E_mod = E_mod.view(x.size(0), self.M, self.K, -1)
                #y = torch.einsum('ij, bikj -> ibk',x,T)
        else:
            E_mod=A
            y = torch.einsum('ij, bkj -> ibk',x,A)
        #y = torch.einsum('bijk, ik -> ibk',T,y)
        #E_mod=torch.tensor(A)
        #y = torch.einsum('ij, bkj -> ibk',x,A)
        #y = F.softmax(y,dim=2)
        #print(E_mod[0,4])
        return(x,y,A,E_mod)


class ClassifierNetwork(nn.Module):
    def __init__(self,input_dim,K,fnet_type):
        super(ClassifierNetwork,self).__init__()
        if fnet_type=='fcnn_dropout':
            self.fnet = FCNN_Dropout(input_dim,K)
        elif fnet_type=='lenet':
            self.fnet = Lenet()
        elif fnet_type=='fcnn_dropout_batchnorm':
            self.fnet = FCNN_Dropout_BatchNorm(input_dim,K)
        elif fnet_type=='linear':
            self.fnet = LinearClassifier(input_dim,K)
        elif fnet_type=='resnet9':
            self.fnet = ResNet9(K)
        elif fnet_type=='resnet18f':
            self.fnet = ResNet18_F(K)
        elif fnet_type=='resnet18':
            self.fnet = ResNet18(K)
        elif fnet_type=='resnet34':
            self.fnet = ResNet34(K)
        else:
            self.fnet = FCNN_Dropout()


    def forward(self,x):
        x_softmax,y = self.fnet(x)
        return(x_softmax,y)





class ConfusionNetwork(nn.Module):
    def __init__(self,M,K,init_method,A_init,flag_instance_dep_modeling):
        super(ConfusionNetwork,self).__init__()
        if init_method=='close_to_identity':
            self.P = nn.Parameter(torch.stack([6*torch.eye(K)-5 for _ in range(M)]), requires_grad=True)
        elif init_method=='mle_based':
            self.P = nn.Parameter(torch.stack([A_init[m] for m in range(M)]), requires_grad=True)
        else:
            print(init_method)
            self.P = nn.Parameter(torch.stack([torch.eye(K) for _ in range(M)]), requires_grad=True)


        self.fc = nn.ModuleList([nn.Linear(K, K * K, bias=False) for i in range(M)])
        self.M=M
        self.K=K
        self.flag_instance_dep_modeling=flag_instance_dep_modeling
##                self.ones =  torch.ones(K)
##                self.zeros = torch.zeros([K, K])
##                self.w = torch.Tensor([])
##                for i in range(K):
##                        '''k = random.randint(0,self.num_classes-1)
##                        while k==i:
##                                k = random.randint(0,self.num_classes-1)'''
##                        temp = self.zeros.clone()
##                        ind = temp[i].add_(self.ones-0.1)
##                        temp = temp+0.1/self.K
##                        self.w = torch.cat([self.w, temp.detach()], 0)
##                for i in range(M):
##                        self.fc[i].weight.data= self.w
##                print(self.w)
    def forward(self,x):
        #A = F.softplus(self.P)
        #A = F.normalize(A,p=1,dim=1)
        A = F.softmax(self.P,dim=1)
        if self.flag_instance_dep_modeling==1:
            E = torch.stack([self.fc[i](x) for i in range(self.M)])
            E = E.view(self.M, x.size(0), self.K, -1)
            E = F.normalize(E, p=2, dim=2)
            #A = F.softmax(self.P,dim=1)
            A_mod= A.view(A.size(0),1,A.size(1),A.size(2))
            T = torch.tensor(E+A_mod)
            T = F.softmax(T, dim=2)
            E_mod = T-A_mod
        else:
            E_mod=A
        #y = torch.einsum('ij, bikj -> ibk',x,T)
        return(A,E_mod)


class CrowdNetwork1(nn.Module):
    def __init__(self,input_dim,M,K,fnet_type,init_method,A_init):
        super(CrowdNetwork1,self).__init__()
        if fnet_type=='fcnn_dropout':
            self.fnet = FCNN_Dropout(input_dim,K)
        elif fnet_type=='lenet':
            self.fnet = Lenet()
        elif fnet_type=='fcnn_dropout_batchnorm':
            self.fnet = FCNN_Dropout_BatchNorm(input_dim,K)
        elif fnet_type=='linear':
            self.fnet = LinearClassifier(input_dim,K)
        elif fnet_type=='resnet9':
            self.fnet = ResNet9(K)
        elif fnet_type=='resnet18':
            self.fnet = ResNet18(K)
        elif fnet_type=='resnet34':
            self.fnet = ResNet34(K)
        else:
            self.fnet = FCNN_Dropout()
        self.Ams = nn.ModuleList([LinearMatrix(K,K) for i in range(M)])
    def forward(self,x):
        x = self.fnet(x)
        Ax = self.A(x)

        #A = F.softmax(self.P,dim=1)
        y = torch.einsum('ij, bkj -> ibk',x,A)
        #y = F.softmax(y,dim=2)
        return(x,y,A)


class LinearMatrix(nn.Module):
    def __init__(self,input_dim,K):
        super(LinearMatrix,self).__init__()
        self.linear = nn.Linear(input_dim,K,bias=False)

    def forward(self,x):
        x=self.linear(x)
        x=F.softmax(x,dim=1)
        return x




class confusion_matrices(nn.Module):
    def __init__(self, device, M, K, init_method='close_to_identity', projection_type='simplex_projection',A_init=[]):
        super(confusion_matrices, self).__init__()
        if init_method=='close_to_identity':
            P = torch.zeros(M, K, K)
            for i in range(M):
                P[i] = torch.eye(K)#6*torch.eye(K)-5#+ P[i]*C
        elif init_method=='from_file':
            H=torch.tensor(np.loadtxt('A_matrix_init.txt',delimiter=','))
            P =torch.zeros(M,K,K)
            for i in range(M):
                P[i] = H[i*K:(i+1)*K,:]#+ P[i]*C
            print(P.size())
        elif init_method=='mle_based':
            P = torch.log(A_init+0.01)
        elif init_method=='dsem_based':
            P = torch.tensor(A_init).float()
        elif init_method=='deviation_from_identity':
            P = -2*torch.ones(M, K, K)
            C = torch.ones(K, K)
            ind = np.diag_indices(C.shape[0])
            C[ind[0], ind[1]] = torch.zeros(C.shape[0])
            self.C = C.to(device)
            self.identity = torch.eye(K).to(device)
        self.register_parameter(name='W', param=nn.parameter.Parameter(P))
        self.W.to(device)
        self.M = M
        self.K = K
        self.projection_type=projection_type

    def forward(self):
        if self.projection_type=='simplex_projection':
            A = F.softplus(self.W)
            A = F.normalize(A.clone(),p=1,dim=1)
        elif self.projection_type=='sigmoid_projection':
            sig = torch.sigmoid(self.W)
            A = torch.zeros((self.M, self.K, self.K))
            for i in range(self.M):
                A[i] = self.identity.detach() + sig[i,:,:]*self.C.detach()
            A = F.normalize(A.clone(), p=1, dim=1)
        elif self.projection_type=='softmax':
            A = F.softmax(self.W,dim=1)
        else:
            A = []

        return A

class confusion_matrices_tracereg(nn.Module):
    def __init__(self, device, M, K, init_method='close_to_identity', A_init=[]):
        super(confusion_matrices_tracereg, self).__init__()
        if init_method=='close_to_identity':
            P = torch.sigmoid(-10*torch.ones(M, K, K))
            C = torch.ones(K, K)
            ind = np.diag_indices(C.shape[0])
            C[ind[0], ind[1]] = torch.zeros(C.shape[0])
            for i in range(M):
                P[i] = torch.eye(K)#+ P[i]*C
        elif init_method=='from_file':
            H=torch.tensor(np.loadtxt('A_matrix_init.txt',delimiter=','))
            P =torch.zeros(M,K,K)
            for i in range(M):
                P[i] = H[i*K:(i+1)*K,:]#+ P[i]*C
            print(P.size())
        elif init_method=='mle_based':
            P = torch.log(A_init+0.01)
        self.register_parameter(name='W', param=nn.parameter.Parameter(P))
        self.W.to(device)
        self.M = M
        self.K = K

    def forward(self):
        A = F.relu(self.W)
        A = F.normalize(A.clone(),p=1,dim=1)
        #A = F.softmax(self.W,dim=1)

        return A

class confusion_matrix(nn.Module):
    def __init__(self, device, K, init=2):
        super(confusion_matrix, self).__init__()
        self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(K, K)))
        self.w.to(device)
        co = torch.ones(K, K)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.to(device)
        self.identity = torch.eye(K).to(device)
        self.K = K


    def forward(self):
        sig = torch.sigmoid(self.w)
        A = self.identity.detach() + sig*self.co.detach()
        A = F.normalize(A.clone(), p=1, dim=0)
        return A

class Lenet(nn.Module):

    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(400, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)


    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        clean = F.softmax(out, 1)


        return clean

#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#        self.conv2_drop = nn.Dropout2d()
#        self.fc1 = nn.Linear(320, 50)
#        self.fc2 = nn.Linear(50, 10)
#
#    def forward(self, x):
#        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#        x = x.view(-1, 320)
#        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
#        x = self.fc2(x)
#        return F.log_softmax(x)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class LinearClassifier(nn.Module):
    def __init__(self,input_dim,K):
        super(LinearClassifier,self).__init__()
        self.linear = nn.Linear(input_dim,K)

    def forward(self,x):
        x=self.linear(x)
        x=F.softmax(x,dim=1)
        return x


class FCNN_Dropout(nn.Module):

    def __init__(self,input_dim,K):
        super(FCNN_Dropout, self).__init__()
        layer_list = []
        layer_list.append(nn.Flatten(start_dim=1))
        layer_list.append(nn.Linear(input_dim, 128))
        layer_list.append(nn.ReLU(inplace=False))
        layer_list.append(nn.Dropout(0.5))
        layer_list.append(nn.Linear(128, K))
        self.layers=nn.Sequential(*layer_list)

    def forward(self,x):
        out = self.layers(x)
        out_softmax = F.softmax(out,dim=1)
        return out_softmax

class SequentialDataModel(nn.Module):

    def __init__(self,num_words,):
        super(SequentialDataModel, self).__init__()
        layer_list = []
        layer_list.append(nn.Embedding(num_words,300))
        layer_list.append(nn.Conv1d(512,5))
        layer_list.append(nn.ReLU(inplace=False))
        layer_list.append(nn.Dropout(0.5))
        layer_list.append(nn.Linear(128, K))
        self.layers=nn.Sequential(*layer_list)

    def forward(self,x):
        out = self.layers(x)
        out = F.softmax(out,dim=1)
        return out

def build_base_model():
    base_model = Sequential()
    base_model.add(Embedding(num_words,
        300,
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=True))
    base_model.add(Conv1D(512, 5, padding="same", activation="relu"))
    base_model.add(Dropout(0.5))
    base_model.add(GRU(50, return_sequences=True))
    base_model.add(TimeDistributed(Dense(N_CLASSES, activation='softmax')))
    base_model.compile(loss='categorical_crossentropy', optimizer='adam')

    return base_model

class FCNN_Dropout_BatchNorm(nn.Module):

    def __init__(self,input_dim,K):
        super(FCNN_Dropout_BatchNorm, self).__init__()
        layer_list = []
        layer_list.append(nn.Flatten(start_dim=1))
        layer_list.append(nn.BatchNorm1d(input_dim,affine=False))
        layer_list.append(nn.Linear(input_dim, 128))
        layer_list.append(nn.ReLU(inplace=False))
        layer_list.append(nn.Dropout(0.5))
        layer_list.append(nn.BatchNorm1d(128,affine=False))
        layer_list.append(nn.Linear(128, K))
        self.layers=nn.Sequential(*layer_list)

    def forward(self,x):
        out = self.layers(x)
        out_softmax = F.softmax(out,dim=1)
        return out_softmax, out

#class FCNN_DropoutNosoftmax(nn.Module):
#
#    def __init__(self,input_dim,K):
#        super(FCNN_DropoutNosoftmax, self).__init__()
#        layer_list = []
#        layer_list.append(nn.Flatten(start_dim=1))
#        layer_list.append(nn.Linear(input_dim, 128))
#        layer_list.append(nn.ReLU(inplace=False))
#        layer_list.append(nn.Dropout(0.5))
#        self.layers=nn.Sequential(*layer_list)
#
#    def forward(self,x):
#        out = self.layers(x)
#        return out


class FCNN_DropoutNosoftmax(nn.Module):

    def __init__(self,input_dim,K):
        super(FCNN_DropoutNosoftmax, self).__init__()
        layer_list = []
        layer_list.append(nn.Flatten(start_dim=1))
        layer_list.append(nn.Linear(input_dim, 128))
        layer_list.append(nn.ReLU(inplace=False))
        layer_list.append(nn.Dropout(0.5))
        layer_list.append(nn.Linear(128, K))
        self.layers=nn.Sequential(*layer_list)

    def forward(self,x):
        out = self.layers(x)
        #out = F.softmax(out,dim=1)
        return out


class FeatureExtractor(nn.Module):

    def __init__(self, base_model='inception'):
        super(FeatureExtractor, self).__init__()
        if base_model== 'vgg16':
            self.model = models.vgg16(pretrained=True)
        else:
            self.model = hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
            self.model.fc = nn.Identity()

    def forward(self, x):
        feature = self.model(x)
        if self.training:
            feature = feature.logits
        return feature


class Weights(nn.Module):

    def __init__(self, n_annotators, weight_type='W', feature_dim=2048, bottleneck_dim=None):
        super(Weights, self).__init__()
        self.weight_type = weight_type
        if self.weight_type == 'W':
            self.weights = nn.Parameter(torch.ones(n_annotators), requires_grad=True)
        elif self.weight_type == 'I':
            if bottleneck_dim is None:
                self.weights = nn.Linear(feature_dim, n_annotators)
            else:
                self.weights = nn.Sequential(nn.ReLU(), nn.Linear(feature_dim, bottleneck_dim), nn.Linear(bottleneck_dim, n_annotators))
        else:
            raise IndexError("weight type must be 'W' or 'I'.")

    def forward(self, feature):
        if self.weight_type == 'W':
            return self.weights
        else:
            return self.weights(feature).view(-1)


class DoctorNet(nn.Module):

    def __init__(self, n_classes, n_annotators, weight_type='W', feature_dim=2048, bottleneck_dim=None, base_model='inception'):
        super(DoctorNet, self).__init__()
        self.feature_extractor = FeatureExtractor(base_model)
        self.annotators = nn.Parameter(torch.stack([torch.randn(feature_dim, n_classes) for _ in range(n_annotators)]), requires_grad=True)
        self.weights = Weights(n_annotators, weight_type, feature_dim, bottleneck_dim)

    def forward(self, x, pred=False, weight=False):
        feature = self.feature_extractor(x)
        decisions = torch.einsum('ik,jkl->ijl', feature, self.annotators)
        weights = self.weights(feature)
        if weight:
            decisions = decisions * weights[None, :, None]
        if pred:
            decisions = torch.sum(decisions, axis=1)
            return decisions
        else:
            return decisions, weights


class CoNAL(nn.Module):
    def __identity_init(self, shape):
        out = np.ones(shape) * 0
        if len(shape) == 3:
            for r in range(shape[0]):
                for i in range(shape[1]):
                    out[r, i, i] = 2
        elif len(shape) == 2:
            for i in range(shape[1]):
                out[i, i] = 2
        return torch.Tensor(out).to(self.device)

    def __init__(self, device, fnet_type, num_annotators, input_dims, num_class, rate=0.5, conn_type='MW', backbone_model=None, user_feature=None
            , common_module='simple', num_side_features=None, nb=None, u_features=None,
            v_features=None, u_features_side=None, v_features_side=None, input_dim=None, emb_dim=None, hidden=None, gumbel_common=False):
        super(CoNAL, self).__init__()
        self.device = device
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        self.gumbel_sigmoid = GumbelSigmoid(temp=0.01)
        K=num_class
        if fnet_type=='fcnn_dropout':
            self.fnet = FCNN_Dropout(input_dims,K)
        elif fnet_type=='lenet':
            self.fnet = Lenet()
        elif fnet_type=='fcnn_dropout_batchnorm':
            self.fnet = FCNN_Dropout_BatchNorm(input_dims,K)
        elif fnet_type=='resnet9':
            self.fnet = ResNet9(K)
        elif fnet_type=='resnet18':
            self.fnet = ResNet18(K)
        elif fnet_type=='resnet18f':
            self.fnet = ResNet18_F(K)
        elif fnet_type=='resnet34':
            self.fnet = ResNet34(K)
        else:
            self.fnet = FCNN_Dropout()

        self.rate = rate
        self.kernel = nn.Parameter(self.__identity_init((num_annotators, num_class, num_class)),
                requires_grad=True)

        self.common_kernel = nn.Parameter(self.__identity_init((num_class, num_class)) ,
                requires_grad=True)

        self.backbone_model = None
        if backbone_model == 'vgg16':
            self.backbone_model = VGG('VGG16').to(self.device)
            self.feature = self.backbone_model.features
            self.classifier = self.backbone_model.classifier
        self.common_module = common_module

        if self.common_module == 'simple':
            com_emb_size = 20
            self.user_feature_vec = torch.from_numpy(user_feature).float().to(self.device)
            self.diff_linear_1 = nn.Linear(input_dims, 128)
            self.diff_linear_2 = nn.Linear(128, com_emb_size)
            self.user_feature_1 = nn.Linear(self.user_feature_vec.size(1), com_emb_size)
            self.bn_instance = torch.nn.BatchNorm1d(com_emb_size, affine=False)
            self.bn_user = torch.nn.BatchNorm1d(com_emb_size, affine=False)
            self.single_weight = nn.Linear(20, 1, bias=False)

    def simple_common_module(self, input):
        instance_difficulty = self.diff_linear_1(input)
        instance_difficulty = self.diff_linear_2(instance_difficulty)

        instance_difficulty = F.normalize(instance_difficulty)
        user_feature = self.user_feature_1(self.user_feature_vec)
        user_feature = F.normalize(user_feature)
        common_rate = torch.einsum('ij,kj->ik', (instance_difficulty, user_feature))
        common_rate = torch.sigmoid(common_rate)
        return common_rate

    def forward(self, input, y=None, mode='train', support=None, support_t=None, idx=None):
        crowd_out = None
        if self.backbone_model:
            cls_out = self.backbone_model(input)
        else:
            #x = input.view(input.size(0), -1)
            #x = self.dropout1(F.relu(self.linear1(x)))
            #x = self.linear2(x)
            cls_out = self.fnet(input)
        if mode == 'train':
            x = input.view(input.size(0), -1)
            if self.common_module == 'simple':
                common_rate = self.simple_common_module(x)
            common_prob = torch.einsum('ij,jk->ik', (cls_out, self.common_kernel))
            indivi_prob = torch.einsum('ik,jkl->ijl', (cls_out, self.kernel))

            crowd_out = common_rate[:, :, None] * common_prob[:, None, :] + (1 - common_rate[:, :, None]) * indivi_prob   # single instance
            crowd_out = crowd_out.transpose(1, 2)
        if self.common_module == 'simple' or mode == 'test':
            return cls_out, crowd_out

class CoNAL_music(nn.Module):
    def __identity_init(self, shape):
        out = np.ones(shape) * 0
        if len(shape) == 3:
            for r in range(shape[0]):
                for i in range(shape[1]):
                    out[r, i, i] = 2
        elif len(shape) == 2:
            for i in range(shape[1]):
                out[i, i] = 2
        return torch.Tensor(out).to(self.device)

    def __init__(self, device, num_annotators, input_dims, num_class, rate=0.5, conn_type='MW', backbone_model=None, user_feature=None
            , common_module='simple', num_side_features=None, nb=None, u_features=None,
            v_features=None, u_features_side=None, v_features_side=None, input_dim=None, emb_dim=None, hidden=None, gumbel_common=False):
        super(CoNAL_music, self).__init__()
        self.device = device
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        self.gumbel_sigmoid = GumbelSigmoid(temp=0.01)

        self.linear1 = nn.Linear(input_dims, 128)

        self.ln1 = nn.Linear(128, 256)
        self.ln2 = nn.Linear(256, 128)

        self.bn = torch.nn.BatchNorm1d(input_dims, affine=False)
        self.bn1 = torch.nn.BatchNorm1d(128, affine=False)

        self.linear2 = nn.Linear(128, num_class)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.rate = rate
        self.kernel = nn.Parameter(self.__identity_init((num_annotators, num_class, num_class)),
                requires_grad=True)

        self.common_kernel = nn.Parameter(self.__identity_init((num_class, num_class)) ,
                requires_grad=True)

        self.backbone_model = None
        if backbone_model == 'vgg16':
            self.backbone_model = VGG('VGG16').to(self.device)
            self.feature = self.backbone_model.features
            self.classifier = self.backbone_model.classifier
        self.common_module = common_module

        if self.common_module == 'simple':
            com_emb_size = 80
            self.user_feature_vec = torch.from_numpy(user_feature).float().to(self.device)
            self.diff_linear_1 = nn.Linear(input_dims, 128)
            self.diff_linear_2 = nn.Linear(128, com_emb_size)
            self.user_feature_1 = nn.Linear(self.user_feature_vec.size(1), com_emb_size)
            self.bn_instance = torch.nn.BatchNorm1d(com_emb_size, affine=False)
            self.bn_user = torch.nn.BatchNorm1d(com_emb_size, affine=False)
            self.single_weight = nn.Linear(20, 1, bias=False)

    def simple_common_module(self, input):
        instance_difficulty = self.diff_linear_1(input)
        instance_difficulty = self.diff_linear_2(instance_difficulty)

        user_feature = self.user_feature_1(self.user_feature_vec)
        user_feature = F.normalize(user_feature)
        common_rate = torch.einsum('ij,kj->ik', (instance_difficulty, user_feature))
        common_rate = torch.sigmoid(common_rate)
        return common_rate

    def forward(self, input, y=None, mode='train', support=None, support_t=None, idx=None):
        crowd_out = None
        if self.backbone_model:
            cls_out = self.backbone_model(input)
        else:
            x = input.view(input.size(0), -1)
            x = self.bn(x)
            x = self.dropout1(F.relu(self.linear1(x)))
            x = self.bn1(x)
            x = self.linear2(x)
            cls_out = torch.nn.functional.softmax(x, dim=1)
        if mode == 'train':
            x = input.view(input.size(0), -1)
            if self.common_module == 'simple':
                common_rate = self.simple_common_module(x)
            elif self.common_module == 'gcn':
                u = list(range(self.num_annotators))
                common_rate, rec_out = self.gae(u, idx, support, support_t)
                common_rate = common_rate.transpose(0, 1)
            common_prob = torch.einsum('ij,jk->ik', (cls_out, self.common_kernel))
            indivi_prob = torch.einsum('ik,jkl->ijl', (cls_out, self.kernel))

            crowd_out = common_rate[:, :, None] * common_prob[:, None, :] + (1 - common_rate[:, :, None]) * indivi_prob   # single instance
            crowd_out = crowd_out.transpose(1, 2)
        if self.common_module == 'simple' or mode == 'test':
            return cls_out, crowd_out

class MAXMIG_left(nn.Module):
    def __init__(self,fnet_type,input_dim,K):
        super(MAXMIG_left, self).__init__()
        if fnet_type=='fcnn_dropout':
            self.fnet = FCNN_Dropout(input_dim,K)
        elif fnet_type=='lenet':
            self.fnet = Lenet()
        elif fnet_type=='fcnn_dropout_batchnorm':
            self.fnet = FCNN_Dropout_BatchNorm(input_dim,K)
        elif fnet_type=='resnet9':
            self.fnet = ResNet9(K)
        elif fnet_type=='resnet18':
            self.fnet = ResNet18(K)
        elif fnet_type=='resnet18f':
            self.fnet = ResNet18_F(K)
        elif fnet_type=='resnet34':
            self.fnet = ResNet34(K)
        else:
            self.fnet = FCNN_Dropout()

    def forward(self, x):
        x = self.fnet(x)
        return x

#class MAXMIG_left(nn.Module):
#    def __init__(self):
#        super(MAXMIG_left, self).__init__()
#        self.linear1 = nn.Linear(8192,128)
#        self.linear2 = nn.Linear(128,8)
#        self.dropout1 = nn.Dropout(0.5)
#
#    def forward(self, x):
#        x = x.resize(x.size()[0],8192)
#        x = self.dropout1(F.relu(self.linear1(x)))
#        x = self.linear2(x)
#
#        return torch.nn.functional.softmax(x,dim=1)

class MAXMIG_left_music(nn.Module):
    def __init__(self):
        super(MAXMIG_left_music, self).__init__()
        self.bn1 = nn.BatchNorm1d(124,affine=False)
        self.linear1 = nn.Linear(124,128)
        self.linear2 = nn.Linear(128,10)
        self.dropout1 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(128,affine=False)

    def forward(self, x):
        x = x.resize(x.size()[0],124)
        x = self.dropout1(F.relu(self.linear1(self.bn1(x))))
        x = self.linear2(self.bn2(x))

        return torch.nn.functional.softmax(x,dim=1)

class MAXMIG_right(nn.Module):
    def __init__(self,device,prior,train_loader,M,K):
        super(MAXMIG_right, self).__init__()
        #self.priority = prior.unsqueeze(1).cuda()
        self.device = device
        self.train_loader = train_loader
        self.priority = prior.to(self.device)
        self.p = nn.Linear(1,2,bias=False)
        for i in range(M):
            m_name = "fc" + str(i+1)
            self.add_module(m_name,nn.Linear(Config.num_classes, Config.num_classes, bias=False))

        self.weights_init(M,K)

    def forward(self, x, left_p, prior = 0, type=0) :

        entity = torch.ones((x.size()[0],1)).to(self.device)
        out = 0
        for name, module in self.named_children():
            if name == 'p':
                continue
            index = int(name[2:])
            #print(module(x[:, index-1, :]))
            out += module(x[:, index-1, :])
        #print("Out:  ",out)
        #priority =  self.p(entity)
        if type == 1 :
            out += torch.log(left_p+0.001) + torch.log(self.priority)
        elif type == 2 :
            out += torch.log(self.priority)
        elif type == 3 :
            out += torch.log(left_p + 0.001)
        return torch.nn.functional.softmax(out,dim=1)

    def weights_init(self,M,K):
        expert_tmatrix = Initial_mats(self.train_loader,M,K)
        for name, module in self.named_children():
            if name == 'p':
                module.weight.data = self.priority
                continue
            index = int(name[2:])
            module.weight.data = torch.log(expert_tmatrix[index - 1] + 0.01)


    def weights_update(self, expert_parameters):
        for name, module in self.named_children():
            if name == 'p':
                continue
            index = int(name[2:])
            module.weight.data = torch.log(expert_parameters[index - 1] + 0.0001)


    def get_prior(self):
        for name, module in self.named_children():
            if name == 'p':
                return module.weight.data


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, revision=True):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        clean = F.softmax(out, 1)

        return clean, out


class ResNet_F(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_F, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.T_revision = nn.Linear(num_classes, num_classes, False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, revision=False):

        correction = self.T_revision.weight

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out_1 = out.view(out.size(0), -1)
        out_2 = self.linear(out_1)
        clean=F.softmax(out_2, 1)
        if revision == True:
            return  clean, correction
        else:
            return  clean, out_2

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

def ResNet18_F(num_classes):
    return ResNet_F(BasicBlock, [2,2,2,2], num_classes)


class ResNet9(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Why does the input size ratchet up to 512 like this??
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                nn.Flatten(),
                nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        out_softmax = F.softmax(out, 1)
        return out_softmax, out

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                    )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
