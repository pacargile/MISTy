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

def defmod(D_in,H1,H2,H3,D_out,xmin,xmax,nntype='LinNet'):
    if nntype == 'CNN':
        return NNmodels.CNN(D_in,H1,H2,D_out,xmin,xmax)
    elif nntype == 'LinNet':
        return NNmodels.LinNet(D_in,H1,H2,H3,D_out,xmin,xmax)
    else:
        return NNmodels.SMLP(D_in,H1,H2,H3,D_out,xmin,xmax)

def readNN(nnpath,nntype='LinNet'):
    # read in the file for the previous run 
    nnh5 = h5py.File(nnpath,'r')

    xmin  = nnh5['xmin'][()]
    xmax  = nnh5['xmax'][()]

    if (nntype == 'SMLP'):
        D_in  = nnh5['model/features.0.weight'].shape[1]
        H1    = nnh5['model/features.0.weight'].shape[0]
        H2    = nnh5['model/features.2.weight'].shape[0]
        H3    = nnh5['model/features.4.weight'].shape[0]
        D_out = nnh5['model/features.6.weight'].shape[0]

    if (nntype == 'LinNet'):
        D_in  = nnh5['model/lin1.weight'].shape[1]
        H1    = nnh5['model/lin1.weight'].shape[0]
        H2    = nnh5['model/lin4.weight'].shape[0]
        H3    = nnh5['model/lin5.weight'].shape[0]
        D_out = nnh5['model/lin6.weight'].shape[0]
    
    if nntype == 'CNN':
        D_in      = nnh5['model/encoder.0.weight'].shape[1]
        H1        = nnh5['model/encoder.0.bias'].shape[0]
        H2        = nnh5['model/encoder.2.bias'].shape[0]
        H3 = None
        D_out     = nnh5['model/encoder.6.bias'].shape[0]
    
    model = defmod(D_in,H1,H2,H3,D_out,xmin,xmax,nntype=nntype)

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
    th5 = h5py.File(self.nnpath,'r')
    
    if self.normed:
        self.ymin = th5['ymin'][()]
        self.ymax = th5['ymax'][()]

    self.nntype = kwargs.get('nntype','LinNet')

    self.model = readNN(self.nnpath,nntype=self.nntype)

  def eval(self,x):
    if isinstance(x,list):
        x = np.asarray(x)
    if len(x.shape) == 1:
        inputD = 1
    else:
        inputD = x.shape[0]

    inputVar = Variable(torch.from_numpy(x).type(dtype)).reshape(inputD,self.model.D_in)
    outpars = self.model(inputVar)
    outpars = outpars.data.numpy().squeeze()

    if self.normed:
        outpars = np.array([self.unnorm(x,ii) for ii,x in enumerate(outpars)])
    return outpars

  def unnorm(self,x,ii):
    return (x + 0.5)*(self.ymax[ii]-self.ymin[ii]) + self.ymin[ii]

class modpred(object):
  """docstring for modpred"""
  def __init__(self, nnpath=None, nntype='LinNet', normed=False):
    super(modpred, self).__init__()
    if nnpath != None:
      self.nnpath = nnpath
    else:
      self.nnpath  = misty.__abspath__+'data/ANN/mistyNN.h5'

    self.anns = ANN(nnpath=self.nnpath,nntype=nntype,normed=normed)

  def pred(self,inpars):
    return self.anns.eval(inpars)