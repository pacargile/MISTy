import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) != 'cpu':
  dtype = torch.cuda.FloatTensor
else:
  dtype = torch.FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import conv1d as tconv1d
from torch.nn.functional import conv_transpose1d as tconv_transpose1d

import jax.numpy as np
from jax import lax
from flax import linen
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

        self.D_in = len(self.label_in)
        self.D_out = len(self.label_out)

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

            self.e_bias0   = nnh5['model/encoder.0.bias'][()]
            self.e_weight0 = nnh5['model/encoder.0.weight'][()]
            self.e_bias2   = nnh5['model/encoder.2.bias'][()]
            self.e_weight2 = nnh5['model/encoder.2.weight'][()]
            self.e_bias4   = nnh5['model/encoder.4.bias'][()]
            self.e_weight4 = nnh5['model/encoder.4.weight'][()]
            self.e_bias6   = nnh5['model/encoder.6.bias'][()]
            self.e_weight6 = nnh5['model/encoder.6.weight'][()]

            self.d_bias0   = nnh5['model/decoder.0.bias'][()]
            self.d_weight0 = nnh5['model/decoder.0.weight'][()]
            self.d_bias2   = nnh5['model/decoder.2.bias'][()]
            self.d_weight2 = nnh5['model/decoder.2.weight'][()]
            self.d_bias4   = nnh5['model/decoder.4.bias'][()]
            self.d_weight4 = nnh5['model/decoder.4.weight'][()]
            self.d_bias6   = nnh5['model/decoder.6.bias'][()]
            self.d_weight6 = nnh5['model/decoder.6.weight'][()]
            
            self.eval = self.evalResNet

        nnh5.close()

    def leaky_relu(self,z):
        '''
        This is the activation function used by default in all our neural networks.
        '''
        return z*(z > 0) + 0.01*z*(z < 0)
            
    def norm(self,x):
        x_np = np.array(x)
        x_scaled = (x_np-self.xmin)/(self.xmax-self.xmin) - 0.5
        return x_scaled

    def unnorm(self,y,ii):
        return (y + 0.5)*(self.ymax[ii]-self.ymin[ii]) + self.ymin[ii]

    def evalSMLP(self,x):
        x_i = self.norm(x)
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

    def elu(self,x):
        return (x >= 0.0) * x + (x < 0.0) * (np.exp(x) - 1.0)

    """

    def sliding_window_view_jax(self, arr, window_shape):
        window_shape = np.asarray(window_shape)

        if arr.ndim != 2:
            raise ValueError("Input array must be 2-dimensional for sliding window view.")

        rows, cols = arr.shape
        win_rows, win_cols = window_shape

        if win_rows > rows or win_cols > cols:
            raise ValueError("Window shape cannot be larger than input array dimensions.")

        # Compute the number of windows along each dimension
        num_windows_rows = rows - win_rows + 1
        num_windows_cols = cols - win_cols + 1

        # Create a list of window views using explicit indexing
        window_views = []
        for i in range(num_windows_rows):
            for j in range(num_windows_cols):
                window = arr[i:i+win_rows, j:j+win_cols]
                window_views.append(window)

        # Stack the window views along a new axis to create the final sliding window view
        window_view = np.stack(window_views, axis=-1)

        return window_view

    def conv1d(self, input, weight, bias=None, stride=1, padding=0):
        # Extract dimensions
        in_channels, input_length = input.shape
        out_channels, _, kernel_size = weight.shape
        
        # Calculate output length
        output_length = (input_length + 2 * padding - kernel_size) // stride + 1
        
        # Add padding to input
        if padding > 0:
            padding_zeros = np.zeros((in_channels, padding))
            input_padded = np.concatenate((padding_zeros, input, padding_zeros), axis=1)
        else:
            input_padded = input
        
        # Reshape input and weight tensors for efficient computation
        # input_reshaped = np.lib.stride_tricks.sliding_window_view(input_padded, (in_channels, kernel_size))
        input_reshaped = self.sliding_window_view_jax(input_padded, (in_channels, kernel_size))
        input_reshaped = input_reshaped[::stride].reshape(output_length, -1)
        weight_reshaped = weight.reshape(out_channels, -1).T
        
        # Perform convolution
        output_reshaped = np.dot(input_reshaped, weight_reshaped)
        
        # Reshape output tensor
        output = output_reshaped.T
        
        # Add bias if provided
        if bias is not None:
            output += bias.reshape(out_channels, 1)
        
        return output

    def conv1d(self, input, weight, bias=None, stride=1, padding=0):
        # Extract dimensions
        in_channels, input_length = input.shape
        out_channels, _, kernel_size = weight.shape
        
        # Calculate output length
        output_length = (input_length + 2 * padding - kernel_size) // stride + 1
        
        # Add padding to input
        if padding > 0:
            pad_widths = [(0, 0), (padding, padding)]
            input_padded = np.pad(input, pad_widths)
        else:
            input_padded = input
        
        # Reshape input and weight tensors for efficient computation
        input_reshaped = np.reshape(input_padded, (1, in_channels, input_length, 1))
        weight_reshaped = np.reshape(weight, (out_channels, in_channels, kernel_size, 1))
        
        # Perform convolution
        output = lax.conv_transpose(input_reshaped, weight_reshaped, strides=(stride,), padding='VALID')
        
        # Reshape output tensor
        output = np.squeeze(output, axis=(0, 3))
        
        # Add bias if provided
        if bias is not None:
            output += np.reshape(bias, (out_channels, 1))
        
        return output

    def conv_transpose1d(self, input, weight, bias=None, stride=1, padding=0, output_padding=0):
        # Extract dimensions
        batch_size, in_channels, input_length = input.shape
        out_channels, _, kernel_size = weight.shape
        
        # Calculate output length
        output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding
        
        # Reshape input and weight tensors for efficient computation
        input_reshaped = input.transpose(0, 2, 1).reshape(batch_size * input_length, in_channels)
        weight_reshaped = weight.reshape(in_channels, -1)
        
        # Perform transpose convolution
        output_reshaped = np.dot(input_reshaped, weight_reshaped.T)
        output_reshaped = output_reshaped.reshape(batch_size, input_length, out_channels)
        
        # Add padding to output if necessary
        if padding > 0:
            output_reshaped = output_reshaped[:, padding:-padding]
        
        # Adjust stride and padding for output length
        stride = stride + output_padding
        
        # Upsample output
        output = np.zeros((batch_size, out_channels, output_length))
        output[:, :, ::stride] = output_reshaped
        
        # Add bias if provided
        if bias is not None:
            output += bias.reshape(1, out_channels, 1)
        
        return output
    """

    def evalResNet(self,x):
        x_i = self.norm(x)
        x_i = x_i.T[np.newaxis,:]

        # encoder
        layer1 = linen.Conv(
            features=128, kernel_size=(1,), padding=1, strides=1).apply(
                {'params': {'kernel': np.transpose(self.e_weight0, (2,1,0)), 'bias': self.e_bias0}}, 
                x_i)
        layer2 = linen.Conv(
            features=256, kernel_size=(3,), padding=1, strides=2).apply(
                {'params': {'kernel': np.transpose(self.e_weight2, (2,1,0)), 'bias': self.e_bias2}}, 
                self.elu(layer1))
        layer3 = linen.Conv(
            features=128, kernel_size=(14,), padding=0, strides=1).apply(
                {'params': {'kernel': np.transpose(self.e_weight4, (2,1,0)), 'bias': self.e_bias4}}, 
                self.elu(layer2))
        layer4 = linen.Conv(
            features=self.D_out, kernel_size=(1,), padding=0, strides=1).apply(
                {'params': {'kernel': np.transpose(self.e_weight6, (2,1,0)), 'bias': self.e_bias6}}, 
                self.elu(layer3))
        
        # layer1 = self.conv1d(x_i,              self.e_weight0, bias=self.e_bias0, stride=1, padding=1)
        # layer2 = self.conv1d(self.elu(layer1), self.e_weight2, bias=self.e_bias2, stride=2, padding=1)
        # layer3 = self.conv1d(self.elu(layer2), self.e_weight4, bias=self.e_bias4, stride=1, padding=0)
        # layer4 = self.conv1d(self.elu(layer3), self.e_weight6, bias=self.e_bias6, stride=1, padding=0)

        # decoder
        layer5 = linen.ConvTranspose(
            features=128, kernel_size=(1,), padding=0, strides=(1,)).apply(
                {'params': {'kernel': np.transpose(self.d_weight0, (2,0,1)), 'bias': self.d_bias0}}, 
                self.elu(layer4))
        layer6 = linen.ConvTranspose(
            features=256, kernel_size=(14,), padding=0, strides=(1,)).apply(
                {'params': {'kernel': np.transpose(self.d_weight2, (2,0,1)), 'bias': self.d_bias2}}, 
                self.elu(layer5))
        layer7 = linen.ConvTranspose(
            features=128, kernel_size=(3,), padding=1, strides=(2,)).apply(
                {'params': {'kernel': np.transpose(self.d_weight4, (2,0,1)), 'bias': self.d_bias4}}, 
                self.elu(layer6))
        y_i = linen.ConvTranspose(
            features=self.D_out, kernel_size=(1,), padding=1, strides=(1,)).apply(
                {'params': {'kernel': np.transpose(self.d_weight6, (2,0,1)), 'bias': self.d_bias6}}, 
                self.elu(layer7))

        # layer5 = self.conv_transpose1d(layer4,           self.d_weight0, bias=self.d_bias0, stride=1, padding=0)
        # layer6 = self.conv_transpose1d(self.elu(layer5), self.d_weight2, bias=self.d_bias2, stride=1, padding=0)
        # layer7 = self.conv_transpose1d(self.elu(layer6), self.d_weight4, bias=self.d_bias4, stride=2, padding=1, output_padding=1)
        # y_i    = self.conv_transpose1d(self.elu(layer7), self.d_weight6, bias=self.d_bias6, stride=1, padding=1, output_padding=0)
        
        # y_i = y_i.T

        if self.normed:
            y = np.array([self.unnorm(yy,ii) for ii,yy in enumerate(y_i)])
        else:
            y = y_i

        return y

class modpred(object):
  """docstring for modpred"""
  def __init__(self, nnpath=None, nntype='LinNet', normed=False, trainage=False):
    super(modpred, self).__init__()
    if nnpath != None:
      self.nnpath = nnpath
    else:
      self.nnpath  = misty.__abspath__+'data/ANN/mistyNN.h5'

    self.trainagebool = trainage

    self.anns = Net(nnpath=self.nnpath,nntype=nntype,normed=normed)

    self.modpararr = ([
        'EEP',
        'initial_Mass',
        'initial_[Fe/H]',
        'initial_[a/Fe]',
        'Mass',
        'log(Age)',
        'log(R)',
        'log(L)',
        'log(Teff)',
        '[Fe/H]',
        '[a/Fe]',
        'log(g)',
        ])

  def pred(self,inpars):
    return self.anns.eval(inpars)

  def getMIST(self,**kwargs):

    eep = kwargs.get('eep',300)
    mass = kwargs.get('mass',1.0)
    feh = kwargs.get('feh',0.0)
    afe = kwargs.get('afe',0.)
    logage = kwargs.get('logage',9.5)

    if self.trainagebool:
        x = np.asarray([logage,mass,feh,afe])
    else:
        x = np.asarray([eep,mass,feh,afe])

    modpred = self.pred(x)
    
    out = {}
    if self.trainagebool:
        out['log(Age)'] = logage
    else:
        out['EEP'] = eep
    out['initial_Mass'] = mass
    out['initial_[Fe/H]'] = feh
    out['initial_[a/Fe]'] = afe

    out_i = {x.decode('utf-8'):modpred[ii] for ii,x in enumerate(self.anns.label_out)}
    out.update(out_i)

    out['log(g)'] = out.pop('log_g')
    out['log(Teff)'] = out.pop('log_Teff')
    out['Mass'] = out.pop('star_mass')
    out['log(R)'] = out.pop('log_R')
    out['log(L)'] = out.pop('log_L')

    if self.trainagebool:
        out['EEP'] = out.pop('EEP')
    else:
        out['log(Age)'] = out.pop('log_age')

    return out
