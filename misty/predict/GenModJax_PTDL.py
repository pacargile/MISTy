# import torch
# from torch import nn
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if str(device) != 'cpu':
#   dtype = torch.cuda.FloatTensor
# else:
#   dtype = torch.FloatTensor
# from torch.autograd import Variable
# import torch.nn.functional as F
# from torch.nn.functional import conv1d as tconv1d
# from torch.nn.functional import conv_transpose1d as tconv_transpose1d

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as np

from flax import nnx

from jax import lax
# from flax import linen
import warnings
import h5py
import time,sys,os,glob
from datetime import datetime

class Net(object):
    def __init__(self, nnpath=None,nntype='SMLP',normed=False):
        self.normed = normed
        self.readNN(nnpath=nnpath,nntype=nntype)

    def readNN(self,nnpath='',nntype='SMLP'):
        # read in normalization info
        nnh5 = h5py.File(nnpath,'r')

        self.label_i = [x.decode('utf-8') for x in nnh5['label_i'][()]]
        self.label_o = [x.decode('utf-8') for x in nnh5['label_o'][()]]

        if self.normed:
            self.norm_i = [nnh5[f'norm_i/{kk}'][()] for kk in self.label_i]
            self.norm_o = [nnh5[f'norm_o/{kk}'][()] for kk in self.label_o]

        # self.label_in  = nnh5['label_i'][()]
        # self.label_out = nnh5['label_o'][()]

        self.D_in = len(self.label_i)
        self.D_out = len(self.label_o)

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

        if (nntype == 'MLP'):
            self.bias1 = np.array(nnh5['model/mlp.lin1.bias'][()])
            self.bias2 = np.array(nnh5['model/mlp.lin2.bias'][()])
            self.bias3 = np.array(nnh5['model/mlp.lin3.bias'][()])
            self.bias4 = np.array(nnh5['model/mlp.lin4.bias'][()])
            self.bias5 = np.array(nnh5['model/mlp.lin5.bias'][()])
            self.bias6 = np.array(nnh5['model/mlp.lin6.bias'][()])

            self.weight1 = np.transpose(np.array(nnh5['model/mlp.lin1.weight'][()]),(1,0))
            self.weight2 = np.transpose(np.array(nnh5['model/mlp.lin2.weight'][()]),(1,0))
            self.weight3 = np.transpose(np.array(nnh5['model/mlp.lin3.weight'][()]),(1,0))
            self.weight4 = np.transpose(np.array(nnh5['model/mlp.lin4.weight'][()]),(1,0))
            self.weight5 = np.transpose(np.array(nnh5['model/mlp.lin5.weight'][()]),(1,0))
            self.weight6 = np.transpose(np.array(nnh5['model/mlp.lin6.weight'][()]),(1,0))

            self.lin1 = nnx.Linear(in_features=self.weight1.shape[0],out_features=self.weight1.shape[1],rngs=nnx.Rngs(0))
            self.lin1.kernel = nnx.Param(value=self.weight1)
            self.lin1.bias = nnx.Param(value=self.bias1)

            self.lin2 = nnx.Linear(in_features=self.weight2.shape[0],out_features=self.weight2.shape[1],rngs=nnx.Rngs(0))
            self.lin2.kernel = nnx.Param(value=self.weight2)
            self.lin2.bias = nnx.Param(value=self.bias2)

            self.lin3 = nnx.Linear(in_features=self.weight3.shape[0],out_features=self.weight3.shape[1],rngs=nnx.Rngs(0))
            self.lin3.kernel = nnx.Param(value=self.weight3)
            self.lin3.bias = nnx.Param(value=self.bias3)

            self.lin4 = nnx.Linear(in_features=self.weight4.shape[0],out_features=self.weight4.shape[1],rngs=nnx.Rngs(0))
            self.lin4.kernel = nnx.Param(value=self.weight4)
            self.lin4.bias = nnx.Param(value=self.bias4)

            self.lin5 = nnx.Linear(in_features=self.weight5.shape[0],out_features=self.weight5.shape[1],rngs=nnx.Rngs(0))
            self.lin5.kernel = nnx.Param(value=self.weight5)
            self.lin5.bias = nnx.Param(value=self.bias5)

            self.lin6 = nnx.Linear(in_features=self.weight6.shape[0],out_features=self.weight6.shape[1],rngs=nnx.Rngs(0))
            self.lin6.kernel = nnx.Param(value=self.weight6)
            self.lin6.bias = nnx.Param(value=self.bias6)
            
            self.mlp = nnx.Sequential(
                self.lin1,
                # nnx.sigmoid,
                nnx.gelu,
                self.lin2,
                # nnx.sigmoid,
                nnx.gelu,
                self.lin3,
                # nnx.sigmoid,
                nnx.gelu,
                self.lin4,
                # nnx.sigmoid,
                nnx.gelu,
                self.lin5,
                # nnx.sigmoid,
                nnx.gelu,
                self.lin6,
            )

            self.eval = self.evalMLP

        nnh5.close()

    def sigmoid(self, a):
        return 1. / (1 + np.exp(-a))

    def evalLinNet(self,x):
        x_i = self.norm(x)

        layer1  = np.einsum('ij,j->i', self.weight1, x_i)                  + self.bias1
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

    def evalMLP(self,x):
        if self.normed:
            x_i = np.zeros(x.shape,dtype=float)
            for ii,n_i in enumerate(self.norm_i):
                x_i = x_i.at[ii].set(1.0 + (x[ii]-n_i[0])/(n_i[2]-n_i[1]))
        else:
            x_i = x

        y_i = self.mlp(x_i)
        
        if self.normed:
            y = np.zeros(y_i.shape,dtype=float)
            for ii,n_i in enumerate(self.norm_o):
                y = y.at[ii].set(((y_i[ii]-1.0)*(n_i[2]-n_i[1])) + n_i[0])
        else:
            y = y_i

        return y


class modpred(object):
    """docstring for modpred"""
    def __init__(self, nnpath=None, nntype='LinNet', normed=False, trainage=False, applyspot=False):
        super(modpred, self).__init__()
        if nnpath != None:
            self.nnpath = nnpath
        else:
            self.nnpath  = misty.__abspath__+'data/ANN/mistyNN.h5'

        self.applyspot = applyspot

        self.anns = Net(nnpath=self.nnpath,nntype=nntype,normed=normed)

        self.modpararr = self.anns.label_o

    def pred(self,inpars):
        return self.anns.eval(inpars)

    def getMIST(self,x):#**kwargs):

        # x = np.array([kwargs.get('age'),kwargs.get('mass'),kwargs.get('feh'),kwargs.get('afe')])        

        modpred = self.pred(x)

        out = {}
        # stick in input labels
        for ii,kk in enumerate(self.anns.label_i):
            out[kk] = x[ii]#kwargs.get(kk)
        
        out_i = {x:modpred[ii] for ii,x in enumerate(self.anns.label_o)}
        out.update(out_i)

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

    def corretpars(self,Told,Rold,gold):
        beta = lax.cond(gold > 4.0, self.berdyugina_beta, lambda x: 0.0, Told)

        # beta = self.berdyugina_beta(Told)
        gamma = self.berdyugina_gamma(Told)
        alfa = 1.0 - beta

        Tnew = (alfa*(Told**4.0) + beta*((Told*gamma)**4.0))**0.25
        Rnew = Rold * (Told/Tnew)**2.0
        gnew = gold * (Rold/Rnew)**2.0

        return (Tnew,Rnew,gnew)

    def berdyugina_beta(self,Teff):
        cond = (Teff > 3098.89) & (Teff < 6101.11)
        # cond = (Teff < 6101.11)
        A = lax.cond(cond, lambda Teff : -1.77514793E-7, lambda Teff : 0.0, Teff)
        B = lax.cond(cond, lambda Teff :  1.63313609E-3, lambda Teff : 0.0, Teff)
        C = lax.cond(cond, lambda Teff : -3.35621302,    lambda Teff : 0.0, Teff)
        return A*Teff*Teff + B*Teff + C

        # a = lax.cond(cond, lambda Teff : 0.4, lambda Teff : 0.0, Teff)
        # b = 1.0
        # c = lax.cond(cond, lambda Teff : 7.5E-3,    lambda Teff : 0.0, Teff)
        # d = lax.cond(cond, lambda Teff : 3500.0,    lambda Teff : 0.0, Teff)

        # return a / (b + np.exp(c * (Teff-d)))        
                
    def berdyugina_gamma(self,Teff):
        return 1.0 - 0.5 * self.berdyugina_beta(Teff)