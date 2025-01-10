import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if str(device) == "cuda:0":
    dtype = torch.cuda.FloatTensor
else:
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps:0")
    dtype = torch.FloatTensor

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,ExponentialLR
from torch.utils.data import DataLoader

import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Pool

from astropy.table import Table,vstack

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import traceback
import numpy as np
from scipy.stats import scoreatpercentile
import warnings
import h5py
import time,sys,os,glob,shutil
from datetime import datetime

from ..utils import NNmodels
from ..utils import readmist_PTDL as readmist
from ..predict import GenMod_PTDL as GenMod
# from ..predict import GenModJax as GenMod

def slicebatch(inlist,N):
    '''
    Function to slice a list into batches of N elements. Last sublist might have < N elements.
    '''
    return [inlist[ii:ii+N] for ii in range(0,len(inlist),N)]

def defmod(D_in,H1,H2,H3,D_out,NNtype='MLP'):
    if NNtype == 'LinNet':
        return NNmodels.LinNet(D_in,H1,H2,H3,D_out)
    elif NNtype == 'CNN':
        return NNmodels.CNN(D_in,H1,H2,H3,D_out)
    else:
        return NNmodels.MLP(D_in,H1,H2,H3,D_out)

class TrainMod(object):
    """docstring for TrainMod"""
    def __init__(self, *arg, **kwargs):
        super(TrainMod, self).__init__()

        print(f'... Start Training Code at {datetime.now()}')
        sys.stdout.flush()

        # turn on/off log plotting
        self.logplot = kwargs.get('logplot',True)

        # taining details
        self.trainper  = kwargs.get('trainper',0.9)
        self.numiters  = int(kwargs.get('numiters',1e+4))
        self.numepochs = kwargs.get('numepochs',10)
        self.batchsize = kwargs.get('batchsize',1000)
        self.nminibatches = kwargs.get('nminibatches',10)
        self.minibatchsize = kwargs.get('minibatchsize',2048)

        # starting learning rate
        self.lr = kwargs.get('lr',1E-4)

        # number of nuerons in each layer
        self.H1 = kwargs.get('H1',256)
        self.H2 = kwargs.get('H2',256)
        self.H3 = kwargs.get('H3',256)

        # create list of in labels and out labels
        if 'label_i' in kwargs:
            self.label_i = kwargs['label_i']
        else:
            self.label_i = ['EEP','initial_mass','initial_[Fe/H]','initial_[a/Fe]']
        if 'label_o'in kwargs:
            self.label_o = kwargs['label_o']
        else:
            self.label_o = ([
                'star_mass',
                'log_L',
                'log_Teff',
                'log_R',
                'log_g',
                'log_age',
                '[Fe/H]',
                '[a/Fe]',
                ])

        # check for user defined ranges for atm models
        defaultparrange = ({
            'EEP':[0.0,1000000.0],
            'initial_mass':[-1.0,10000.0],
            'initial_[Fe/H]':[-10.0,10.0],
            'initial_[a/Fe]':[-10.0,10.0],
            })
        self.parrange = kwargs.get('parrange',defaultparrange)

        # self.eeprange  = kwargs.get('eep',[0.0,1000000.0])
        # self.massrange = kwargs.get('mass',[-1.0,10000.0])
        # self.fehrange  = kwargs.get('feh',[-10.0,10.0])
        # self.aferange  = kwargs.get('afe',[-10.0,10.0])

        self.restartfile = kwargs.get('restartfile',None)
        if self.restartfile is not None:
            print('... Restarting File: {0}'.format(self.restartfile))

        # output hdf5 file name
        self.outfilename = kwargs.get('output','TESTOUT.h5')
        
        # path to MIST tracks
        self.mistpath  = kwargs.get('mistpath',None)

        # type of NN to train
        self.NNtype = kwargs.get('NNtype','LinNet')

        # the output predictions are normed
        self.norm = kwargs.get('norm',True)
        
        print(f'... Running with normalized labels: {self.norm}')

        # use eepprior in training
        self.eepprior = kwargs.get('eepprior',False)
        if self.eepprior:
            print('... Running with EEP prior on training sampler')

        print('... Running Training on Device: {}'.format(device))

        # initialzie class to pull models
        print('... Pulling a first set of models for test set')
        print('... Reading {0:.2f} of grid for test models from {1}'.format(1.0-self.trainper,self.mistpath))
        sys.stdout.flush()
        test_mistmods = readmist.ReadMIST(
            mistpath=self.mistpath,
            label_i=self.label_i,
            label_o=self.label_o,
            norm=False,
            returntorch=False,
            type='test',
            trainpercentage=self.trainper,
            parrange=self.parrange,
            eepprior=False,
            )
        print(f'... Total number of test models: {len(test_mistmods)}')        
        test_dataloader = DataLoader(test_mistmods, batch_size=len(test_mistmods),shuffle=True)

        testdata = next(iter(test_dataloader))

        self.datacond_in  = np.array([np.where(test_mistmods.columns == val)[0][0] for val in self.label_i],dtype=int)
        self.datacond_out = np.array([np.where(test_mistmods.columns == val)[0][0] for val in self.label_o],dtype=int)
        
        # self.datacond_in  = np.in1d(test_mistmods.columns,self.label_i)
        # self.datacond_out = np.in1d(test_mistmods.columns,self.label_o)
        
        self.test_labelsin  = testdata[:,self.datacond_in]
        self.test_labelsout = testdata[:,self.datacond_out]

        print('... Finished reading in test set of models')
        
        sys.stdout.flush()

        # determine normalization values
        # self.xmin = np.array([test_mistmods.minmax[x][0] 
        #     for x in self.label_i])
        # self.xmax = np.array([test_mistmods.minmax[x][1] 
        #     for x in self.label_i])

        # self.ymin = np.array([test_mistmods.minmax[x][0]
        #     for x in self.label_o])
        # self.ymax = np.array([test_mistmods.minmax[x][1] 
        #     for x in self.label_o])

        if self.norm:
            self.normfactor = test_mistmods.normfactor
        else:
            self.normfactor = None

        # D_in is input dimension
        # D_out is output dimension
        self.D_in = len(self.label_i)
        self.D_out = len(self.label_o)

        print('... Din: {}, Dout: {}'.format(self.D_in,self.D_out))
        print('... Input Labels: {}'.format(self.label_i))
        print('... Output Labels: {}'.format(self.label_o))

        # initialize the output file
        with h5py.File('{0}'.format(self.outfilename),'w') as outfile_i:
            try:
                outfile_i.create_dataset('testlabels_in',
                    data=self.test_labelsin)
                outfile_i.create_dataset('testlabels_out',
                    data=self.test_labelsout)
                outfile_i.create_dataset('label_i',
                    data=np.array([x.encode("ascii", "ignore") for x in self.label_i]))
                outfile_i.create_dataset('label_o',
                    data=np.array([x.encode("ascii", "ignore") for x in self.label_o]))
                if self.norm:
                    for kk in self.label_i:                
                        outfile_i.create_dataset(f'norm_i/{kk}',data=np.array(self.normfactor[kk]))
                    for kk in self.label_o:                
                        outfile_i.create_dataset(f'norm_o/{kk}',data=np.array(self.normfactor[kk]))
                
                # outfile_i.create_dataset('xmin',data=np.array(self.xmin))
                # outfile_i.create_dataset('xmax',data=np.array(self.xmax))
                # outfile_i.create_dataset('ymin',data=np.array(self.ymin))
                # outfile_i.create_dataset('ymax',data=np.array(self.ymax))
            except:
                print('!!! PROBLEM WITH WRITING TO HDF5 !!!')
                raise

        print('... Finished Init')
        sys.stdout.flush()

    def __call__(self):
        '''
        call instance so that train_pixel can be called with multiprocessing
        and still have all of the class instance variables

        '''
        try:
            return self.train_mod()
        except Exception as e:
            traceback.print_exc()
            print()
            raise e

    def run(self):
        '''
        function to actually run the training on models

        '''
        # start total timer
        tottimestart = datetime.now()

        print('Starting Training at {0}'.format(tottimestart))
        sys.stdout.flush()

        net = self()

        tottimeend = datetime.now()

        print('Finished Training at {0} ({1})'.format(tottimeend,tottimeend-tottimestart))
        if type(net[0]) == type(None):
            return net

    def train_mod(self):
        '''
        function to train the network
        '''
        # start a timer
        starttime = datetime.now()

        if str(device) == 'cuda':
            # determine if this is running within mp
            if len(multiprocessing.current_process()._identity) > 0:
                torch.cuda.set_device(multiprocessing.current_process()._identity[0]-1)

            print('Running on GPU: {0}/{1}'.format(
                torch.cuda.current_device()+1,
                torch.cuda.device_count(),
                ))

        # determine if user wants to start from old file, or
        # create a new ANN model
        if self.restartfile is not None:
            # create a model
            if os.path.isfile(self.restartfile):
                print('Restarting from File: {0} with NNtype: {1}'.format(self.restartfile,self.NNtype))
                sys.stdout.flush()
                model = GenMod.readNN(self.restartfile,nntype=self.NNtype)
                # model = GenMod.Net(nnpath=self.restartfile,nntype=self.NNtype,normed=True)
            else:
                print('Could Not Find Restart File, Creating a New NN model')
                sys.stdout.flush()
                model = defmod(self.D_in,self.H1,self.H2,self.H3,self.D_out,NNtype=self.NNtype)        
        else:
            # initialize the model
            print('Running New NN with NNtype: {0}'.format(self.NNtype))
            sys.stdout.flush()
            model = defmod(self.D_in,self.H1,self.H2,self.H3,self.D_out,NNtype=self.NNtype)

        print('Model Arch:')
        print(model)

        # set up model to start training
        model.to(device)

        train_mistmods = readmist.ReadMIST(
            mistpath=self.mistpath,
            label_i=self.label_i,
            label_o=self.label_o,
            norm=self.norm,
            returntorch=True,
            type='train',
            trainpercentage=self.trainper,
            parrange=self.parrange,
            eepprior=self.eepprior,
            )
        valid_mistmods = readmist.ReadMIST(
            mistpath=self.mistpath,
            label_i=self.label_i,
            label_o=self.label_o,
            norm=self.norm,
            returntorch=True,
            type='valid',
            trainpercentage=self.trainper,
            parrange=self.parrange,
            eepprior=self.eepprior,
            )

        if self.eepprior:
            batchsampler_train = readmist.EEPBatchSampler(train_mistmods, self.batchsize)
            batchsampler_valid = readmist.EEPBatchSampler(valid_mistmods, self.batchsize)
            train_dataloader = DataLoader(train_mistmods, batch_sampler=batchsampler_train)
            valid_dataloader = DataLoader(valid_mistmods, batch_sampler=batchsampler_valid)
        else:
            train_dataloader = DataLoader(train_mistmods, batch_size=self.batchsize, shuffle=True)
            valid_dataloader = DataLoader(valid_mistmods, batch_size=self.batchsize, shuffle=True)

        nbatches = len(train_dataloader)
        numtrain = nbatches * self.batchsize
        
        print(f'... Number of epochs: {self.numepochs}')
        print(f'... Number of training steps: {self.numiters}')
        print(f'... Number of models in each batch: {self.batchsize}')
        print(f'... Number of batches: {nbatches}')
        print(f'... Total Number of training/validation data: {numtrain}')

        try:
            if shutil.which('free') is not None:
                total_memory, used_memory, free_memory = map(
                    int, os.popen('free -t -g').readlines()[-1].split()[1:])
                print('--- Current Memory Usage Before Epoch 1: {0} GB'.format(used_memory))
            else:
                total_memory = 0
                used_memory = 0
                free_memory = 0
        except:
            pass

        # initialize the loss function
        # loss_fn = torch.nn.MSELoss()
        # loss_fn = torch.nn.MSELoss(reduction='mean')
        loss_fn = torch.nn.HuberLoss(reduction='mean')
        # loss_fn = torch.nn.SmoothL1Loss()
        # loss_fn = torch.nn.L1Loss()

        # initialize the optimizer
        learning_rate = self.lr

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=1E-5,decoupled_weight_decay=True)
        # optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,fused=True)

        # initialize the scheduler to adjust the learning rate
        # scheduler = StepLR(optimizer,10000,gamma=0.5,verbose=False)
        # scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.1)
        scheduler = ExponentialLR(optimizer,gamma=1-(5E-8))

        fig1,ax1 = plt.subplots(nrows=len(self.label_i),ncols=1,figsize=(5,15),constrained_layout=True)
        fig2,ax2 = plt.subplots(nrows=len(self.label_o),ncols=1,figsize=(5,15),constrained_layout=True)
        fig3,ax3 = plt.subplots(nrows=3,ncols=1,figsize=(8,8),constrained_layout=True)

        ax3[0].set_ylabel('log(L1 Loss per model)')
        ax3[0].set_xlim(0,self.numiters)
        axins1 = ax3[0].inset_axes([0.75, 0.75, 0.225, 0.225])
        llim = int(0.8 * self.numiters)
        axins1.set_xlim(llim,self.numiters)
        for lab_i in (axins1.get_xticklabels() + axins1.get_yticklabels()):
            lab_i.set_fontsize(5)

        ax3[1].set_ylabel('log(Std Residual)')
        ax3[1].set_xlim(0,self.numiters)

        ax3[2].set_xlabel('Iteration')
        ax3[2].set_ylabel('log(|Med Residual|)')
        ax3[2].set_xlim(0,self.numiters)

        axins2 = ax3[2].inset_axes([0.75, 0.75, 0.225, 0.225])
        axins2.set_xlim(llim,self.numiters)

        for lab_i in (axins2.get_xticklabels() + axins2.get_yticklabels()):
            lab_i.set_fontsize(5)
        
        # cycle through epochs
        for epoch_i,(traindata,validdata) in enumerate(zip(train_dataloader,valid_dataloader)):
            epochtime = datetime.now()                

            print(f'... Pulling {traindata.shape[0]} Training Models for Epoch: {epoch_i+1}')
            sys.stdout.flush()

            # initiate counter
            current_loss = np.inf
            iter_arr = []
            training_loss =[]
            validation_loss = []
            medres_loss = []
            stdres_loss = []

            train_labelsin  = traindata[:,self.datacond_in]
            train_labelsout = traindata[:,self.datacond_out]

            valid_labelsin  = validdata[:,self.datacond_in]
            valid_labelsout = validdata[:,self.datacond_out]

            for ii,kk in enumerate(self.label_i):
                ax1[ii].hist(train_mistmods.unnormf(train_labelsin[:,ii],kk), bins=25, alpha=0.5, histtype='stepfilled')
                ax1[ii].set_xlabel(kk)
            fig1.savefig('plts/{0}_trainingset_T.png'.format(self.outfilename.replace('.h5','')),dpi=150)

            for ii,kk in enumerate(self.label_o):
                ax2[ii].hist(train_mistmods.unnormf(train_labelsout[:,ii],kk), bins=25, alpha=0.5, histtype='stepfilled')
                ax2[ii].set_xlabel(kk)
            fig2.savefig('plts/{0}_trainingset_P.png'.format(self.outfilename.replace('.h5','')),dpi=150)

            # ax1[0,0].hist(train_mistmods.unnormf(train_labelsin[:,0],'EEP'),           bins=25, alpha=0.5, histtype='stepfilled',label=epoch_i+1)
            # ax1[0,1].hist(train_mistmods.unnormf(train_labelsin[:,1],'initial_mass'),  bins=25, alpha=0.5, histtype='stepfilled')
            # ax1[1,0].hist(train_mistmods.unnormf(train_labelsin[:,2],'initial_[Fe/H]'),bins=25, alpha=0.5, histtype='stepfilled')
            # ax1[1,1].hist(train_mistmods.unnormf(train_labelsin[:,3],'initial_[a/Fe]'),bins=25, alpha=0.5, histtype='stepfilled')

            # ax1[0,0].legend()
            # ax1[0,0].set_xlabel('EEP')
            # ax1[0,1].set_xlabel('Mass_i')
            # ax1[1,0].set_xlabel('[Fe/H]_i')
            # ax1[1,1].set_xlabel('[a/Fe]_i')

            # fig1.savefig('plts/{0}_trainingset_T.png'.format(self.outfilename.replace('.h5','')),dpi=150)

            # if self.norm:
            #     ax2[0,0].hist(train_mistmods.unnormf(train_labelsout[:,0],'star_mass'),bins=25, alpha=0.5, histtype='stepfilled',label=epoch_i+1)
            #     ax2[0,1].hist(train_mistmods.unnormf(train_labelsout[:,1],'log_L'),    bins=25, alpha=0.5, histtype='stepfilled')
            #     ax2[1,0].hist(train_mistmods.unnormf(train_labelsout[:,2],'log_Teff'), bins=25, alpha=0.5, histtype='stepfilled')
            #     ax2[1,1].hist(train_mistmods.unnormf(train_labelsout[:,3],'log_R'),    bins=25, alpha=0.5, histtype='stepfilled')
            #     ax2[2,0].hist(train_mistmods.unnormf(train_labelsout[:,4],'log_g'),    bins=25, alpha=0.5, histtype='stepfilled')
            #     ax2[2,1].hist(train_mistmods.unnormf(train_labelsout[:,5],'log_age'),  bins=25, alpha=0.5, histtype='stepfilled')
            #     ax2[3,0].hist(train_mistmods.unnormf(train_labelsout[:,6],'[Fe/H]'),   bins=25, alpha=0.5, histtype='stepfilled')
            #     ax2[3,1].hist(train_mistmods.unnormf(train_labelsout[:,7],'[a/Fe]'),   bins=25, alpha=0.5, histtype='stepfilled')
            # else:
            #     ax2[0,0].hist(train_labelsout[:,0], bins=25, alpha=0.5, histtype='stepfilled',label=epoch_i+1)
            #     ax2[0,1].hist(train_labelsout[:,1], bins=25, alpha=0.5, histtype='stepfilled')
            #     ax2[1,0].hist(train_labelsout[:,2], bins=25, alpha=0.5, histtype='stepfilled')
            #     ax2[1,1].hist(train_labelsout[:,3], bins=25, alpha=0.5, histtype='stepfilled')
            #     ax2[2,0].hist(train_labelsout[:,4], bins=25, alpha=0.5, histtype='stepfilled')
            #     ax2[2,1].hist(train_labelsout[:,5], bins=25, alpha=0.5, histtype='stepfilled')
            #     ax2[3,0].hist(train_labelsout[:,6], bins=25, alpha=0.5, histtype='stepfilled')
            #     ax2[3,1].hist(train_labelsout[:,7], bins=25, alpha=0.5, histtype='stepfilled')
                

            # ax2[0,0].legend()
            # ax2[0,0].set_xlabel('Mass')
            # ax2[0,1].set_xlabel('log(L)')
            # ax2[1,0].set_xlabel('log(Teff)')
            # ax2[1,1].set_xlabel('log(R)')
            # ax2[2,0].set_xlabel('log(g)')
            # ax2[2,1].set_xlabel('log(Age)')
            # ax2[3,0].set_xlabel('[Fe/H]')
            # ax2[3,1].set_xlabel('[a/Fe]')

            # fig2.savefig('plts/{0}_trainingset_P.png'.format(self.outfilename.replace('.h5','')),dpi=150)

            # create tensor for input training labels
            X_train_labels = train_labelsin
            X_train_Tensor = Variable(X_train_labels.type(dtype))
            X_train_Tensor = X_train_Tensor.to(device)

            # create tensor of output training labels
            Y_train_labels = train_labelsout
            Y_train_Tensor = Variable(Y_train_labels.type(dtype), requires_grad=False)
            Y_train_Tensor = Y_train_Tensor.to(device)

            # create tensor for input training labels
            X_valid_labels = valid_labelsin
            X_valid_Tensor = Variable(X_valid_labels.type(dtype), requires_grad=False)
            X_valid_Tensor = X_valid_Tensor.to(device)

            # create tensor of output training labels
            Y_valid_labels = valid_labelsout
            Y_valid_Tensor = Variable(Y_valid_labels.type(dtype), requires_grad=False)
            Y_valid_Tensor = Y_valid_Tensor.to(device)

            print('... Pulling Training & Validation Took {0}'.format(datetime.now()-epochtime))

            try:
                if shutil.which('free') is not None:            
                    total_memory, used_memory, free_memory = map(
                        int, os.popen('free -t -g').readlines()[-1].split()[1:])
                    print('--- Current Memory Usage Before Iteration 1: {0} GB'.format(used_memory))
                else:
                    total_memory = 0
                    used_memory = 0
                    free_memory = 0
            except:
                pass

            for iter_i in range(1,int(self.numiters)+1):
                
                model.train()

                itertime = datetime.now()

                # perm = torch.randperm(len(X_train_Tensor))
                # if str(device) != 'cpu':
                #     perm = perm.to(device)

                # for t in range(self.nminibatches):
                #     steptime = datetime.now()

                #     idx = perm[t * self.minibatchsize : (t+1) * self.minibatchsize]

                #     optimizer.zero_grad()
                #     Y_pred_train_Tensor = model(X_train_Tensor[idx])
                                        
                #     # Compute and print loss.
                #     loss = loss_fn(Y_pred_train_Tensor, Y_train_Tensor[idx])

                #     # Backward pass: compute gradient of the loss with respect to model parameters
                #     loss.backward()

                #     # Calling the step function on an Optimizer makes an update to its parameters
                #     optimizer.step()
                
                
                Y_pred_train_Tensor = model(X_train_Tensor)
                                    
                # Compute and print loss.
                loss = loss_fn(Y_pred_train_Tensor, Y_train_Tensor)

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_data = loss.detach().data.item()
                # adjust the optimizer lr
                LR = scheduler.get_last_lr()[0]

                # evaluate the validation set
                if (iter_i % 1000 == 0) | (iter_i == 1) | (iter_i == int(self.numiters)):

                    print('--> Testing the model @ {}:'.format(iter_i))
                    print('      Input Labels [min / max]:')
                    minval = X_train_Tensor.min(axis=0)[0].tolist()
                    maxval = X_train_Tensor.max(axis=0)[0].tolist()
                    for ii,kk in enumerate(self.label_i):
                        print(f'{kk} -> {minval[ii]:.5f} / {maxval[ii]:.5f}')
                    print('      Output Labels [min / max]  (Pred - Truth: min, max, med):')
                    minval = Y_train_Tensor.min(axis=0)[0].tolist()
                    maxval = Y_train_Tensor.max(axis=0)[0].tolist()
                    f = (Y_pred_train_Tensor - Y_train_Tensor)
                    fmin = f.abs().min(axis=0)[0].tolist()
                    fmax = f.abs().max(axis=0)[0].tolist()
                    fmed = f.abs().median(axis=0)[0].tolist()
                    for ii,kk in enumerate(self.label_o):
                        print(f'{kk} -> {minval[ii]:.5f} / {maxval[ii]:.5f} ({fmin[ii]:.4f}, {fmax[ii]:.4f}, {fmed[ii]:.4f})')
                    
                    if self.logplot:
                        model.eval()

                        loss_valid = 0
                        medres = 0
                        stdres = 0

                        Y_pred_valid_Tensor = model(X_valid_Tensor)                        
                        loss_valid += loss_fn(Y_pred_valid_Tensor, Y_valid_Tensor)
                        loss_valid_data = loss_valid.detach().data.item()

                        # if (iter_i % 100 == 0) and (iter_i != 0) and (j == 0):
                        #     print('--> Testing the model @ {}:'.format(iter_i))
                        #     print('      Input Labels [min / max]:')
                        #     print(X_valid_Tensor[idx].min(axis=0)[0].tolist(),' / ',X_valid_Tensor[idx].max(axis=0)[0].tolist())
                        #     print('      Difference in normalized tensors (Pred - Truth):')
                        #     f = Y_pred_valid_Tensor - Y_valid_Tensor[idx]
                        #     print('      Min:')
                        #     print(['{0:.4f}'.format(x) for x in f.abs().min(axis=0)[0].tolist()])
                        #     print('      Max:')
                        #     print(['{0:.4f}'.format(x) for x in f.abs().max(axis=0)[0].tolist()])
                        #     print('      Median:')
                        #     print(['{0:.4f}'.format(x) for x in f.abs().median(axis=0)[0].tolist()])
                        # print('      Training Labels:')
                        # print(Y_valid_Tensor[idx][:3])
                        # print('      Predicted Labels:')
                        # print(Y_pred_valid_Tensor[:3])

                        residual = torch.abs(Y_pred_valid_Tensor-Y_valid_Tensor)
                        medres_i,stdres_i = float(residual.median()),float(residual.std())
                        if medres_i > medres:
                            medres = medres_i
                        if stdres_i > stdres:
                            stdres = stdres_i

                #     loss_valid /= nbatches


                        iter_arr.append(iter_i)
                        training_loss.append(loss_data)
                        validation_loss.append(loss_valid_data)
                        stdres_loss.append(stdres)
                        medres_loss.append(medres)
                    else:
                        loss_valid_data = np.nan
                        
                if iter_i % 1000 == 0.0:
                    if self.logplot:
                        ax3[0].plot(iter_arr,np.log10(training_loss),ls='-',lw=0.5,alpha=0.5,c='C0',label='Training')
                        ax3[0].plot(iter_arr,np.log10(validation_loss),ls='-',lw=0.5,alpha=0.5,c='C3',label='Validation')
                        # ax3[0].legend(loc='upper center')

                        # plot last 10% of learning curve in inset
                        mask = np.array(iter_arr) >= llim
                        axins1.plot(np.array(iter_arr)[mask],np.log10(np.array(training_loss)[mask]),ls='-',lw=0.5,alpha=0.5,c='C0')
                        axins1.plot(np.array(iter_arr)[mask],np.log10(np.array(validation_loss)[mask]),ls='-',lw=0.5,alpha=0.5,c='C3')
                        try:
                            ul = max([max(np.log10(np.array(validation_loss)[mask])),max(np.log10(np.array(training_loss)[mask]))]) + 1E-5
                            ll = min([min(np.log10(np.array(validation_loss)[mask])),min(np.log10(np.array(training_loss)[mask]))]) - 1E-5
                            axins1.set_ylim(ll - 0.50 * (ul-ll), ul + 0.50 * (ul-ll))
                        except ValueError:
                            pass

                        ax3[1].plot(iter_arr,np.log10(stdres_loss),ls='-',lw=0.5,alpha=0.5,c='C4',label='std')

                        ax3[2].plot(iter_arr,np.log10(medres_loss),ls='-',lw=0.5,alpha=0.5,c='C2',label='median')

                        # plot last 10% of learning curve in inset
                        axins2.plot(np.array(iter_arr)[mask],np.log10(medres_loss)[mask],ls='-',lw=1.0,alpha=1.0,c='C2')                                   
                        try:
                            ul = max(np.log10(medres_loss)[mask]) + 1E-5
                            ll = min(np.log10(medres_loss)[mask]) - 1E-5
                            axins2.set_ylim(ll - 0.50 * (ul-ll), ul + 0.50 * (ul-ll))
                        except ValueError:
                            pass

                        fig3.savefig('plts/{0}_loss.png'.format(self.outfilename.replace('.h5','')),dpi=150)

                    try:

                        if shutil.which('free') is not None:            
                            total_memory, used_memory, free_memory = map(
                                int, os.popen('free -t -g').readlines()[-1].split()[1:])
                        else:
                            total_memory = 0
                            used_memory = 0
                            free_memory = 0                            
                    except:
                        pass
                    print(
                        f'--> Ep: {int(epoch_i+1)} '
                        f'-- Iter: {int(iter_i)}/{self.numiters} '
                        f'-- Time/iter: {datetime.now()-itertime} '
                        f'-- Time: {datetime.now()} '
                        f'-- T Loss: {loss_data:.4f} '
                        f'-- V Loss: {loss_valid_data:.4f} '
                        f'-- log(LR): {np.log10(LR):.5f} '
                        # f'-- Mem Used: {used_memory} GB'
                    )
                    sys.stdout.flush()                      


                # # check if network has converged
                # if np.abs(np.nan_to_num(loss_valid_data)-np.nan_to_num(current_loss))/np.abs(loss_valid_data) < 0.01:
                #     # start counter
                #     cc  = cc + 1

                # if cc == 100:
                #     print(loss_valid_data,current_loss)
                #     current_loss = loss_valid_data
                #     break

                
            # After Each Epoch, write network to output HDF5 file to save progress
            with h5py.File('{0}'.format(self.outfilename),'r+') as outfile_i:
                for kk in model.state_dict().keys():
                    try:
                        del outfile_i['model/{0}'.format(kk)]
                    except KeyError:
                        pass

                    try:
                        outfile_i.create_dataset(
                            'model/{0}'.format(kk),
                            data=model.state_dict()[kk].cpu().numpy(),
                            compression='gzip')
                    except:
                        print('Problem with {0}'.format(kk))
                        raise
            try:
                if shutil.which('free') is not None:            
                    total_memory, used_memory, free_memory = map(
                        int, os.popen('free -t -g').readlines()[-1].split()[1:])
                    print('--- Final Memory Usage: {0} GB'.format(used_memory))
                else:
                    total_memory = 0
                    used_memory = 0
                    free_memory = 0
                
            except:
                pass

            print('Finished Epoch {0} @ {1} ({2}) -- Mem Used: {3} GB'.format(epoch_i+1, datetime.now(),datetime.now() - epochtime, used_memory))
            
            # check if this is last epoch, if so break loop
            if epoch_i+1 == self.numepochs:
                break
            scheduler.step()
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)
            torch.cuda.empty_cache()

        print('Finished training model, took: {0}'.format(
            datetime.now()-starttime))
        sys.stdout.flush()

        return [model, optimizer, datetime.now()-starttime]
