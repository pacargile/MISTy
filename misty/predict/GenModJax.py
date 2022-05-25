import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) != 'cpu':
  dtype = torch.cuda.FloatTensor
else:
  dtype = torch.FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F

import jax.numpy as np
import warnings
import h5py
import time,sys,os,glob
from datetime import datetime

class Net(object):
    def __init__(self, nnpath='',nntype='SMLP',normed=False):
        self.normed = normed
        self.readNN(nnpath=nnpath,nntype=nntype)

    def readNN(self,nnpath='',nntype='SMLP'):
        # read in the file for the previous run 
        nnh5 = h5py.File(nnpath,'r')

        self.xmin  = nnh5['xmin'][()]
        self.xmax  = nnh5['xmax'][()]

        self.ymin  = nnh5['ymin'][()]
        self.ymax  = nnh5['ymax'][()]

        self.label_in  = nnh5['label_i'][()]
        self.label_out = nnh5['label_o'][()]

        if (nntype == 'SMLP'):
            self.bias0 = nnh5['model/features.0.bias'][()]
            self.bias2 = nnh5['model/features.2.bias'][()]
            self.bias4 = nnh5['model/features.4.bias'][()]
            self.bias6 = nnh5['model/features.6.bias'][()]

            self.weight0 = nnh5['model/features.0.weight'][()]
            self.weight2 = nnh5['model/features.2.weight'][()]
            self.weight4 = nnh5['model/features.4.weight'][()]
            self.weight6 = nnh5['model/features.6.weight'][()]

            self.eval = self.evalSMLP

        if (nntype == 'LinNet'):
            self.bias1 = nnh5['model/lin1.bias'][()]
            self.bias2 = nnh5['model/lin2.bias'][()]
            self.bias3 = nnh5['model/lin3.bias'][()]
            self.bias4 = nnh5['model/lin4.bias'][()]
            self.bias5 = nnh5['model/lin5.bias'][()]
            self.bias6 = nnh5['model/lin6.bias'][()]

            self.weight1 = nnh5['model/lin1.weight'][()]
            self.weight2 = nnh5['model/lin2.weight'][()]
            self.weight3 = nnh5['model/lin3.weight'][()]
            self.weight4 = nnh5['model/lin4.weight'][()]
            self.weight5 = nnh5['model/lin5.weight'][()]
            self.weight6 = nnh5['model/lin6.weight'][()]

            self.eval = self.evalLinNet

        if nntype == 'ResNet':
            raise IOError("ResNet not implemented yet with JAX")

        nnh5.close()

    def leaky_relu(self,z):
        '''
        This is the activation function used by default in all our neural networks.
        '''
        return z*(z > 0) + 0.01*z*(z < 0)
     
    def encode(self,x):
        x_np = np.array(x)
        x_scaled = (x_np-self.xmin)/(self.xmax-self.xmin) - 0.5
        return x_scaled

    def unnorm(self,y,ii):
        return (y + 0.5)*(self.ymax[ii]-self.ymin[ii]) + self.ymin[ii]

    def evalSMLP(self,x):
        x_i = self.encode(x)
        layer1  = np.einsum('ij,j->i', self.weight0, x_i) + self.bias0
        layer2  = np.einsum('ij,j->i', self.weight2, self.leaky_relu(layer1)) + self.bias2
        layer3  = np.einsum('ij,j->i', self.weight4, self.leaky_relu(layer2)) + self.bias4
        y_i     = np.einsum('ij,j->i', self.weight6, self.leaky_relu(layer3)) + self.bias6

        if self.normed:
            y = np.array([self.unnorm(yy,ii) for ii,yy in enumerate(y_i)])
        else:
            y = y_i

        return y

    def sigmoid(self, a):
        return 1. / (1 + np.exp(-a))

    def evalLinNet(self,x):
        x_i = self.encode(x)

        layer1  = np.einsum('ij,j->i', self.weight1, x_i) + self.bias1
        layer2  = np.einsum('ij,j->i', self.weight2, self.sigmoid(layer1)) + self.bias2
        layer3  = np.einsum('ij,j->i', self.weight3, self.sigmoid(layer2)) + self.bias3
        layer4  = np.einsum('ij,j->i', self.weight4, self.sigmoid(layer3)) + self.bias4
        layer5  = np.einsum('ij,j->i', self.weight5, self.sigmoid(layer4)) + self.bias5
        y_i     = np.einsum('ij,j->i', self.weight6, self.sigmoid(layer5)) + self.bias6

        if self.normed:
            y = np.array([self.unnorm(yy,ii) for ii,yy in enumerate(y_i)])
        else:
            y = y_i

        return y

class modpred(object):
  """docstring for modpred"""
  def __init__(self, nnpath=None, nntype='SMLP', normed=False):
    super(modpred, self).__init__()
    if nnpath != None:
      self.nnpath = nnpath
    else:
      self.nnpath  = misty.__abspath__+'data/ANN/mistyNN.h5'

    self.anns = Net(nnpath=self.nnpath,nntype=nntype,normed=normed)

    self.modpararr = ([
        'log(Age)',
        'initial_Mass',
        'initial_[Fe/H]',
        'initial_[a/Fe]',
        'Mass',
        'log(R)',
        'log(L)',
        'log(Teff)',
        '[Fe/H]',
        '[a/Fe]',
        'log(g)',
        'EEP',])

  def pred(self,inpars):
    return self.anns.eval(inpars)

  def getMIST(self,logage=9.5, mass=1.0, feh=0.0, afe=0.0, **kwargs):
    x = np.asarray([logage,mass,feh,afe])

    modpred = self.pred(x)
    # output: 'star_mass', 'log_L', 'log_Teff', log_R', 'log_g', '[Fe/H]', '[a/Fe]', 'EEP'

    # logTeff = 0.25 * (modpred[1] - 2.0 * modpred[2]) + np.log10(5772.0)
    # logg    = np.log10(modpred[0]) - 2.0 * modpred[2] + 4.4374

    out = {}
    out['log_age'] = logage
    out['initial_masss'] = mass
    out['initial_[Fe/H]'] = feh
    out['initial_[a/Fe]'] = afe

    for ii,x in enumerate(self.anns.label_out):
        out[x] = modpred[ii]

    # out = [logage,mass,feh,afe,modpred[5],modpred[0],modpred[3],modpred[1],modpred[2],modpred[6],modpred[7],modpred[4]]
    return out
