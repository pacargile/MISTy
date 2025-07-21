import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) != 'cpu':
  dtype = torch.cuda.FloatTensor
else:
  dtype = torch.FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import warnings
import h5py
import time,sys,os,glob
from datetime import datetime

from ..utils import NNmodels

def defmod(D_in,H1,H2,H3,D_out,nntype='MLP'):
    if nntype == 'LinNet':
        return NNmodels.LinNet(D_in,H1,H2,H3,D_out)
    elif nntype == 'CNN':
        return NNmodels.CNN(D_in,H1,H2,H3,D_out)
    elif nntype == 'GenSel':
        return NNmodels.StellarPredictor(input_dim=D_in, output_dim=D_out, latent_dim=H1, latent_steps=H2)
    else:
        return NNmodels.MLP(D_in,H1,H2,H3,D_out)


def readNN(nnpath,nntype='MLP',D_in=None,H1=None,H2=None,H3=None,D_out=None):
    # read in the file for the previous run 
    nnh5 = h5py.File(nnpath,'r')

    if nntype == 'MLP':
        D_in  = nnh5['model/mlp.lin1.weight'].shape[1]
        H1    = nnh5['model/mlp.lin1.weight'].shape[0]
        H2    = nnh5['model/mlp.lin4.weight'].shape[0]
        H3    = nnh5['model/mlp.lin5.weight'].shape[0]
        D_out = nnh5['model/mlp.lin9.weight'].shape[0]
    elif nntype == 'CNN':
        pass
    
    model = defmod(D_in,H1,H2,H3,D_out,nntype=nntype)

    model.D_in = D_in
    model.H1 = H1
    model.H2 = H2
    model.H3 = H3
    model.D_out = D_out

    newmoddict = {}
    for kk in nnh5['model'].keys():
        nparr = nnh5['model'][kk][()]
        torarr = torch.from_numpy(nparr).type(dtype)
        newmoddict[kk] = torarr    
    model.load_state_dict(newmoddict)
    model.eval()
    nnh5.close()
    return model

class ANN(object):
    """docstring for ANN"""
    def __init__(self, nnpath=None,**kwargs):
        super(ANN, self).__init__()

        self.verbose = kwargs.get('verbose',False)

        if nnpath != None:
          self.nnpath = nnpath
        else:
          self.nnpath  = misty.__abspath__+'data/ANN/mistyNN.h5'

        self.normed = kwargs.get('normed',False)

        if self.verbose:
          print('... Reading in {0}'.format(self.nnpath))

        self.nntype = kwargs.get('nntype','LinNet')

        D_in = kwargs.get('D_in',None)
        H1 = kwargs.get('H1',None)
        H2 = kwargs.get('H2',None)
        H3 = kwargs.get('H3',None)
        D_out = kwargs.get('D_out',None)

        self.model = readNN(self.nnpath,nntype=self.nntype, 
                            D_in=D_in,H1=H1,H2=H2,H3=H3,D_out=D_out)


        # read in normalization info
        with h5py.File(self.nnpath,'r') as th5:

            self.label_i = np.array([x.decode('utf-8') for x in th5['label_i'][()]])
            self.label_o = np.array([x.decode('utf-8') for x in th5['label_o'][()]])

            if self.normed:
                self.norm_i = [th5[f'norm_i/{kk}'][()] for kk in self.label_i]
                self.norm_o = [th5[f'norm_o/{kk}'][()] for kk in self.label_o]
        
    def eval(self,x):
        
        # read in x array and check if it is one set of pars or an array of pars
        if isinstance(x,list):
            x = np.asarray(x)

        # make a copy so that the input x array isn't changed in place
        x_i = np.copy(x)

        if len(x.shape) == 1:
            inputD = 1
            if self.normed:
                for ii,n_i in enumerate(self.norm_i):
                    # x[ii] = ((x[ii]-1.0)*(n_i[2]-n_i[1])) + n_i[0]
                    x_i[ii] = 1.0 + (x_i[ii]-n_i[0])/(n_i[2]-n_i[1]) 
        else:
            inputD = x.shape[0]
            if self.normed:
                for ii,n_i in enumerate(self.norm_i):
                    # x[:,ii] = ((x[:,ii]-1.0)*(n_i[2]-n_i[1])) + n_i[0]
                    x_i[:,ii] = 1.0 + (x_i[:,ii]-n_i[0])/(n_i[2]-n_i[1]) 

        inputX = Variable(torch.from_numpy(x_i).type(dtype)).reshape(inputD,self.model.D_in)
        outputY = self.model(inputX)
        y = outputY.data.numpy().squeeze()

        if self.normed:
            if len(x.shape) == 1:
                for ii,n_i in enumerate(self.norm_o):
                    y[ii] = ((y[ii]-1.0)*(n_i[2]-n_i[1])) + n_i[0]
            else:
                for ii,n_i in enumerate(self.norm_o):
                    y[:,ii] = ((y[:,ii]-1.0)*(n_i[2]-n_i[1])) + n_i[0]
        return y

        # if self.normed:
        #     outpars = np.array([self.unnorm(x,ii) for ii,x in enumerate(outpars)])
        # return outpars

    # def unnorm(self,x,ii):
    #     return (x + 0.5)*(self.ymax[ii]-self.ymin[ii]) + self.ymin[ii]

class modpred(object):
    """docstring for modpred"""
    def __init__(self, nnpath=None, nntype='LinNet', **kwargs):
        super(modpred, self).__init__()
        
        if nnpath != None:
            self.nnpath = nnpath
        else:
            self.nnpath  = misty.__abspath__+'data/ANN/mistyNN.h5'

        self.applyspot = kwargs.get('applyspot',False)
        self.normed = kwargs.get('normed',False)

        self.D_in = kwargs.get('D_in',None)
        self.H1 = kwargs.get('H1',None)
        self.H2 = kwargs.get('H2',None)
        self.H3 = kwargs.get('H3',None)
        self.D_out = kwargs.get('D_out',None)

        self.anns = ANN(nnpath=self.nnpath,nntype=nntype,normed=self.normed,
                        D_in=self.D_in,H1=self.H1,H2=self.H2,H3=self.H3,D_out=self.D_out)

        self.modpararr = self.anns.label_o
        
      
    def pred(self,inpars):
        return self.anns.eval(inpars)
    
    def getMIST(self,pars):#**kwargs):

        # x = np.array([kwargs.get('age'),kwargs.get('mass'),kwargs.get('feh'),kwargs.get('afe')])        

        # x = np.array([kwargs.get(kk) for kk in self.anns.label_i])
        
        # make copy of input array so that the code doesn't change inplace
        pars = np.copy(pars)
        
        modpred = self.pred(pars)
    
        out = {}
        
        # stick in input labels
        for ii,kk in enumerate(self.anns.label_i):
            if len(pars.shape) == 1:
                out[kk] = pars[ii] #kwargs.get(kk)
            else:
                out[kk] = pars[:,ii]
        
        if len(pars.shape) == 1:
            out_i = {y:modpred[ii] for ii,y in enumerate(self.anns.label_o)}
            out.update(out_i)
        else:
            out_i = {y:modpred[:,ii] for ii,y in enumerate(self.anns.label_o)}
            out.update(out_i)

        
        # out_i = {x:modpred[ii] for ii,x in enumerate(self.anns.label_o)}
        # out.update(out_i)
        
        if 'log_age' in out.keys():
            out['log(Age)'] = out.pop('log_age')
        if 'log_g' in out.keys():
            out['log(g)'] = out.pop('log_g')
        if 'log_Teff' in out.keys():
            out['log(Teff)'] = out.pop('log_Teff')
        if 'star_mass' in out.keys():
            out['Mass'] = out.pop('star_mass')
        if 'log_R' in out.keys():
            out['log(R)'] = out.pop('log_R')
        if 'log_L' in out.keys():
            out['log(L)'] = out.pop('log_L')

        return out
