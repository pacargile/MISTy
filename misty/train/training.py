import torch
from torch import nn

# try:
#     cudabool = torch.has_cuda
# except:
#     cudabool = False
# try:
#     mpsbool = torch.has_mps
# except:
#     mpsbool = False

# if cudabool:
#     device = torch.device('cuda:0')
# # elif mpsbool:
# #     device = torch.device('mps')
# else:
#     device = torch.device('cpu')

# if device == 'cuda:0':
#     dtype = torch.cuda.FloatTensor
# else:
#     dtype = torch.FloatTensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) != 'cpu':
  dtype = torch.cuda.FloatTensor
else:
  dtype = torch.FloatTensor

from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,ExponentialLR

import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Pool

from astropy.table import Table,vstack

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import traceback
import numpy as np
import warnings
import h5py
import time,sys,os,glob,shutil
from datetime import datetime

from ..utils import NNmodels, radam, readmist
from ..predict import GenMod
# from ..predict import GenModJax as GenMod

def slicebatch(inlist,N):
    '''
    Function to slice a list into batches of N elements. Last sublist might have < N elements.
    '''
    return [inlist[ii:ii+N] for ii in range(0,len(inlist),N)]

def defmod(D_in,H1,H2,H3,D_out,xmin,xmax,NNtype='SMLP'):
    if NNtype == 'ResNet':
        return NNmodels.ResNet(D_in,H1,H2,D_out,xmin,xmax)
    elif NNtype == 'LinNet':
        return NNmodels.LinNet(D_in,H1,H2,H3,D_out,xmin,xmax)
    else:
        return NNmodels.SMLP(D_in,H1,H2,H3,D_out,xmin,xmax)

class TrainMod(object):
    """docstring for TrainMod"""
    def __init__(self, *arg, **kwargs):
        super(TrainMod, self).__init__()

        # number of models to train on
        if 'numtrain' in kwargs:
            self.numtrain = kwargs['numtrain']
        else:
            self.numtrain = 20000

        if 'numtest' in kwargs:
            self.numtest = kwargs['numtest']
        else:
            self.numtest = int(0.1*self.numtrain)

        if 'numsteps' in kwargs:
            self.numsteps = kwargs['numsteps']
        else:
            self.numsteps = int(1e+4)

        if 'numepochs' in kwargs:
            self.numepochs = kwargs['numepochs']
        else:
            self.numepochs = 1

        if 'batchsize' in kwargs:
            self.batchsize = kwargs['batchsize']
        else:
            self.batchsize = self.numtrain

        # number of nuerons in each layer
        if 'H1' in kwargs:
            self.H1 = kwargs['H1']
        else:
            self.H1 = 256

        if 'H2' in kwargs:
            self.H2 = kwargs['H2']
        else:
            self.H2 = 256

        if 'H3' in kwargs:
            self.H3 = kwargs['H3']
        else:
            self.H3 = 256

        # set if user wants to train on age or on EEP, default is EEP
        self.trainagebool = kwargs.get('trainage',False)
        print('... Training on Age: {}'.format(self.trainagebool))

        # create list of in labels and out labels
        if 'label_i' in kwargs:
            self.label_i = kwargs['label_i']
        else:
            if self.trainagebool:
                self.label_i = ['log_age','initial_mass','initial_[Fe/H]','initial_[a/Fe]']
            else:
                self.label_i = ['EEP','initial_mass','initial_[Fe/H]','initial_[a/Fe]']
        if 'label_o'in kwargs:
            self.label_o = kwargs['label_o']
        else:
            if self.trainagebool:
                self.label_o = ['star_mass','log_L','log_Teff','log_R','log_g','[Fe/H]','[a/Fe]','EEP']
            else:
                self.label_o = ['star_mass','log_age','log_L','log_Teff','log_R','log_g','[Fe/H]','[a/Fe]']

        # check for user defined ranges for atm models
        self.eeprange  = kwargs.get('eep',None)
        self.massrange = kwargs.get('mass',None)
        self.FeHrange  = kwargs.get('FeH',None)
        self.aFerange  = kwargs.get('aFe',None)

        self.restartfile = kwargs.get('restartfile',False)
        if self.restartfile is not False:
            print('... Restarting File: {0}'.format(self.restartfile))

        # output hdf5 file name
        self.outfilename = kwargs.get('output','TESTOUT.h5')
        
        # path to MIST tracks
        self.mistpath  = kwargs.get('mistpath',None)

        # type of NN to train
        self.NNtype = kwargs.get('NNtype','LinNet')

        # the output predictions are normed
        self.norm = kwargs.get('norm',False)

        # use eepprior in training
        self.eepprior = kwargs.get('eepprior',False)

        # starting learning rate
        self.lr = kwargs.get('lr',1E-4)

        # user defined test set
        self.testset = kwargs.get('testset',None)

        # user defined training set
        self.trainset = kwargs.get('trainingset',None)

        # user defined validation set
        self.validset = kwargs.get('validset',None)

        print('... Running Training on Device: {}'.format(device))

        # initialzie class to pull models
        print('... Pulling a first set of models for test set')
        print('... Reading {0} test models from {1}'.format(self.numtest,self.mistpath))
        self.mistmods = readmist.readmist(mistpath=self.mistpath)
        sys.stdout.flush()

        # pull a quick set of test models to determine general properties
        if self.testset is None:
            mod_test = self.mistmods.pullmod(
                self.numtest,
                norm=False,
                eepprior=False,
                eep=self.eeprange,mass=self.massrange,feh=self.FeHrange,afe=self.aFerange)
        else:
            mod_test = self.mistmods.readmod(self.testset,
                                             norm=False)

        if self.trainagebool:
            # switch EEP <-> log(Age) so that we can train on age
            self.testlabels_i = mod_test['label_i']
            self.testlabels   = mod_test['label_i'].copy()
            self.testlabels[...,0] = mod_test['log_age']
            del mod_test['log_age']
        else:
            self.testlabels = mod_test['label_i'].copy()

        self.mod_test = Table()
        for x in self.label_o:
            self.mod_test[x] = mod_test[x]

        print('... Finished reading in test set of models')

        # determine normalization values
        self.xmin = np.array([self.mistmods.minmax[x][0] 
            for x in self.label_i])
        self.xmax = np.array([self.mistmods.minmax[x][1] 
            for x in self.label_i])

        self.ymin = np.array([self.mistmods.minmax[x][0]
            for x in self.label_o])
        self.ymax = np.array([self.mistmods.minmax[x][1] 
            for x in self.label_o])

        # D_in is input dimension
        # D_out is output dimension
        self.D_in = len(self.label_i)
        self.D_out = len(self.label_o)

        print('... Din: {}, Dout: {}'.format(self.D_in,self.D_out))
        print('... Input Labels: {}'.format(self.label_i))
        print('... Output Labels: {}'.format(self.label_o))

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
        if self.restartfile is not False:
            # create a model
            if os.path.isfile(self.restartfile):
                print('Restarting from File: {0} with NNtype: {1}'.format(self.restartfile,self.NNtype))
                sys.stdout.flush()
                model = GenMod.readNN(self.restartfile,nntype=self.NNtype)
                # model = GenMod.Net(nnpath=self.restartfile,nntype=self.NNtype,normed=True)
            else:
                print('Could Not Find Restart File, Creating a New NN model')
                sys.stdout.flush()
                model = defmod(self.D_in,self.H1,self.H2,self.H3,self.D_out,
                    self.xmin,self.xmax,NNtype=self.NNtype)        
        else:
            # initialize the model
            print('Running New NN with NNtype: {0}'.format(self.NNtype))
            sys.stdout.flush()
            model = defmod(self.D_in,self.H1,self.H2,self.H3,self.D_out,
                self.xmin,self.xmax,NNtype=self.NNtype)

        # set up model to start training
        model.to(device)

        # initialize the output file
        with h5py.File('{0}'.format(self.outfilename),'w') as outfile_i:
            try:
                outfile_i.create_dataset('testpred',
                    data=np.array(self.mod_test),compression='gzip')
                outfile_i.create_dataset('testlabels',
                    data=self.testlabels,compression='gzip')
                outfile_i.create_dataset('label_i',
                    data=np.array([x.encode("ascii", "ignore") for x in self.label_i]))
                outfile_i.create_dataset('label_o',
                    data=np.array([x.encode("ascii", "ignore") for x in self.label_o]))
                outfile_i.create_dataset('xmin',data=np.array(self.xmin))
                outfile_i.create_dataset('xmax',data=np.array(self.xmax))
                outfile_i.create_dataset('ymin',data=np.array(self.ymin))
                outfile_i.create_dataset('ymax',data=np.array(self.ymax))
            except:
                print('!!! PROBLEM WITH WRITING TO HDF5 !!!')
                raise
        
        # number of batches
        nbatches = self.numtrain // self.batchsize

        print('... Number of epochs: {}'.format(self.numepochs))
        print('... Number of training steps: {}'.format(self.numsteps))
        print('... Number of models in each batch: {}'.format(self.batchsize))
        print('... Number of batches: {}'.format(nbatches))

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
        
        # cycle through epochs
        for epoch_i in range(int(self.numepochs)):
            epochtime = datetime.now()
            print('... Pulling {0} Training Models for Epoch: {1}'.format(self.numtrain,epoch_i+1))
            sys.stdout.flush()

            # initiate counter
            current_loss = np.inf
            iter_arr = []
            training_loss =[]
            validation_loss = []
            medres_loss = []
            maxres_loss = []

            # pull training data
            if self.trainset is None:
                mod_t = self.mistmods.pullmod(
                    self.numtrain,
                    norm=self.norm,
                    eepprior=self.eepprior,                
                    excludelabels=self.testlabels,
                    eep=self.eeprange,mass=self.massrange,feh=self.FeHrange,afe=self.aFerange)
            else:
                mod_t = self.mistmods.readmod(self.trainset,
                                             norm=self.norm)

            fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8,8),constrained_layout=True)

            ax[0].scatter(mod_t['label_i'][:,0],mod_t['label_i'][:,1],marker='.',c='C0',s=5,alpha=0.05)
            ax[1].scatter(mod_t['label_i'][:,2],mod_t['label_i'][:,3],marker='.',c='C0',s=5,alpha=0.05)

            ax[0].set_xlabel('EEP')
            ax[0].set_ylabel('Mass_i')
            
            ax[1].set_xlabel('[Fe/H]_i')
            ax[1].set_ylabel('[a/Fe]_i')

            fig.savefig('{0}_trainingset_T_epoch{1}.png'.format(self.outfilename.replace('.h5',''),epoch_i+1),dpi=150)
            plt.close(fig)

            fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(8,8),constrained_layout=True)

            ax[0].scatter(mod_t['log_Teff'],mod_t['log_g'],marker='.',c='C3',s=5,alpha=0.05)
            ax[1].scatter(mod_t['log_L'],mod_t['log_R'],marker='.',c='C3',s=5,alpha=0.05)
            ax[2].scatter(mod_t['log_age'],mod_t['[Fe/H]'],marker='.',c='C3',s=5,alpha=0.05)

            ax[0].set_xlabel('log(Teff)')
            ax[0].set_ylabel('log(g)')
            
            ax[1].set_xlabel('log(L)')
            ax[1].set_ylabel('log(R)')

            ax[2].set_xlabel('log(Age)')
            ax[2].set_ylabel('[Fe/H]')

            fig.savefig('{0}_trainingset_P_epoch{1}.png'.format(self.outfilename.replace('.h5',''),epoch_i+1),dpi=150)
            plt.close(fig)

            if self.trainagebool:
                # switch EEP <-> log(Age) so that we can train on age
                trainlabels_i      = mod_t['label_i']
                trainlabels        = mod_t['label_i'].copy()
                if self.norm:
                    unnorm_logage_t = (mod_t['log_age'] + 0.5)*(self.mistmods.minmax['log_age'][1]-self.mistmods.minmax['log_age'][0]) + self.mistmods.minmax['log_age'][0]
                else:
                    unnorm_logage_t = mod_t['log_age']
                trainlabels[...,0] = unnorm_logage_t
                del mod_t['log_age']
            else:
                trainlabels = mod_t['label_i'].copy()

            # create tensor for input training labels
            X_train_labels = trainlabels
            X_train_Tensor = Variable(torch.from_numpy(X_train_labels).type(dtype))
            X_train_Tensor = X_train_Tensor.to(device)

            # create tensor of output training labels
            Y_train = np.array([mod_t[x] for x in self.label_o]).T
            Y_train_Tensor = Variable(torch.from_numpy(Y_train).type(dtype), requires_grad=False)
            Y_train_Tensor = Y_train_Tensor.to(device)

            # pull validataion data
            if self.validset is None:
                mod_v = self.mistmods.pullmod(
                    self.numtrain,
                    norm=self.norm,
                    eepprior=self.eepprior,
                    excludelabels=np.array(list(self.testlabels)+list(trainlabels)),
                    eep=self.eeprange,mass=self.massrange,feh=self.FeHrange,afe=self.aFerange)
            else:
                mod_v = self.mistmods.readmod(self.validset,
                                             norm=self.norm)

            if self.trainagebool:
                # switch EEP <-> log(Age) so that we can train on age
                validlabels_i      = mod_v['label_i']
                validlabels        = mod_v['label_i'].copy()
                if self.norm:
                    unnorm_logage_v = (mod_v['log_age'] + 0.5)*(self.mistmods.minmax['log_age'][1]-self.mistmods.minmax['log_age'][0]) + self.mistmods.minmax['log_age'][0]
                else:
                    unnorm_logage_v = mod_v['log_age']
                validlabels[...,0] = unnorm_logage_v
                del mod_v['log_age']
            else:
                validlabels = mod_v['label_i'].copy()

            # create tensor for input validation labels
            X_valid_labels = validlabels
            X_valid_Tensor = Variable(torch.from_numpy(X_valid_labels).type(dtype))
            X_valid_Tensor = X_valid_Tensor.to(device)

            # create tensor of output validation labels
            Y_valid = np.array([mod_v[x] for x in self.label_o]).T
            Y_valid_Tensor = Variable(torch.from_numpy(Y_valid).type(dtype), requires_grad=False)
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

            # initialize the loss function
            loss_fn = torch.nn.MSELoss(reduction='sum')
            # loss_fn = torch.nn.SmoothL1Loss(reduction='sum')
            # loss_fn = torch.nn.KLDivLoss(size_average=False)
            # loss_fn = torch.nn.L1Loss(reduction = 'sum')

            # initialize the optimizer
            learning_rate = self.lr

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
            # optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate, eps=1E-5, weight_decay=1E-3)
            # we adopt rectified Adam for the optimization
            # optimizer = radam.RAdam(
            #     [p for p in model.parameters() if p.requires_grad==True], lr=learning_rate)

            # initialize the scheduler to adjust the learning rate
            # scheduler = StepLR(optimizer,10000,gamma=0.5,verbose=False)
            # scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.1)
            scheduler = ExponentialLR(optimizer,gamma=1-(5E-7),verbose=False)

            cc = 0
            for iter_i in range(int(self.numsteps)):
                model.train()

                itertime = datetime.now()

                perm = torch.randperm(self.numtrain)

                if str(device) != 'cpu':
                    perm = perm.to(device)

                for t in range(nbatches):
                    steptime = datetime.now()

                    idx = perm[t * self.batchsize : (t+1) * self.batchsize]

                    optimizer.zero_grad()
                    Y_pred_train_Tensor = model(X_train_Tensor[idx])
                    
                    # Compute and print loss.
                    loss = loss_fn(Y_pred_train_Tensor, Y_train_Tensor[idx])

                    # Backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()

                    # Calling the step function on an Optimizer makes an update to its parameters
                    optimizer.step()

                loss_data = loss.detach().data.item()
                # adjust the optimizer lr
                LR = scheduler.get_last_lr()[0]
                scheduler.step()


                # evaluate the validation set
                if iter_i % 500 == 0:
                    model.eval()

                    perm_valid = torch.randperm(self.numtrain)
                    if str(device) != 'cpu':
                        perm_valid = perm_valid.to(device)

                    loss_valid = 0
                    medres = 0
                    maxres = 0
                    for j in range(nbatches):
                        idx = perm[t * self.batchsize : (t+1) * self.batchsize]

                        Y_pred_valid_Tensor = model(X_valid_Tensor[idx])                        
                        loss_valid += loss_fn(Y_pred_valid_Tensor, Y_valid_Tensor[idx])

                        if (iter_i % 500 == 0) and (iter_i != 0) and (j == 0):
                            print('--> Testing the model @ {}:'.format(iter_i))
                            print('      Input Labels:')
                            print(X_valid_Tensor[idx][:3])
                            print('      Training Labels:')
                            print(Y_valid_Tensor[idx][:3])
                            print('      Predicted Labels:')
                            print(Y_pred_valid_Tensor[:3])

                        residual = torch.abs(Y_pred_valid_Tensor-Y_valid_Tensor[idx])
                        medres_i,maxres_i = float(residual.median()),float(residual.max())
                        if medres_i > medres:
                            medres = medres_i
                        if maxres_i > maxres:
                            maxres = maxres_i

                    loss_valid /= nbatches

                    loss_valid_data = loss_valid.detach().data.item()

                    iter_arr.append(iter_i)
                    training_loss.append(loss_data)
                    validation_loss.append(loss_valid_data)
                    medres_loss.append(medres)
                    maxres_loss.append(maxres)

                    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(8,8))
                    ax[0].plot(iter_arr,np.log10(training_loss)-np.log10(self.numtrain),ls='-',lw=0.5,alpha=1.0,c='C0',label='Training')
                    ax[0].plot(iter_arr,np.log10(validation_loss)-np.log10(self.numtrain),ls='-',lw=0.5,alpha=1.0,c='C3',label='Validation')
                    ax[0].legend(loc='upper center')
                    ax[0].set_ylabel('log(L1 Loss per model)')

                    axins = ax[0].inset_axes([0.8, 0.8, 0.15, 0.15])
                    # plot last 10% of learning curve in inset
                    xlimbig = ax[0].get_xlim()
                    llim = xlimbig[1] - 0.1 * (xlimbig[1]-xlimbig[0])
                    mask = iter_arr >= llim
                    axins.plot(iter_arr[mask],np.log10(training_loss[mask])-np.log10(self.numtrain),ls='-',lw=0.5,alpha=1.0,c='C0')
                    axins.plot(iter_arr[mask],np.log10(validation_loss[mask])-np.log10(self.numtrain),ls='-',lw=0.5,alpha=1.0,c='C3')
                    axins.set_xlim(llim,xlimbig[1])

                    ax[1].plot(iter_arr,np.log10(maxres_loss),ls='-',lw=0.5,alpha=1.0,c='C4',label='max')
                    ax[1].set_ylabel('log(|Max Residual|)')

                    ax[2].plot(iter_arr,np.log10(medres_loss),ls='-',lw=0.5,alpha=1.0,c='C2',label='median')
                    ax[2].set_xlabel('Iteration')
                    ax[2].set_ylabel('log(|Med Residual|)')

                    fig.savefig('{0}_loss_epoch{1}.png'.format(self.outfilename.replace('.h5',''),epoch_i+1),dpi=150)
                    plt.close(fig)

                    if iter_i % 500 == 0.0:
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

                        print (
                            '--> Ep: {0:d} -- Iter {1:d}/{2:d} -- Time/iter: {3} -- Time: {4} -- T Loss: {5:.4f} -- V Loss: {6:.4f} -- LR: {7} -- Mem Used: {8} GB'.format(
                            int(epoch_i+1),int(iter_i+1),int(self.numsteps), datetime.now()-itertime, datetime.now(), loss_data, loss_valid_data, LR, used_memory)
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


        print('Finished training model, took: {0}'.format(
            datetime.now()-starttime))
        sys.stdout.flush()

        return [model, optimizer, datetime.now()-starttime]


