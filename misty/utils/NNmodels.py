import torch
from torch import nn
from collections import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if str(device) == "cuda:0":
    dtype = torch.cuda.FloatTensor
else:
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps:0")
    dtype = torch.FloatTensor

from torch.autograd import Variable

print('Runing on device: {0}'.format(device))

# import numpy as np
import jax.numpy as np
import warnings
import time,sys,os,glob


# simple multi-layer perceptron model
class SMLP(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out, xmin, xmax):
        super(SMLP, self).__init__()

        self.xmin = xmin
        self.xmax = xmax

        self.features = nn.Sequential(
            nn.Linear(D_in, H1),
            nn.LeakyReLU(),
            nn.Linear(H1, H2),
            nn.LeakyReLU(),
            nn.Linear(H2, H3),
            nn.LeakyReLU(),
            nn.Linear(H3, D_out),
        )
    
    def encode(self,x):
        # convert x into numpy to do math
        x_np = x.data.cpu().numpy()
        xout = (x_np-self.xmin)/(self.xmax-self.xmin) - 0.5
        return Variable(torch.from_numpy(xout).type(dtype))

    def forward(self, x):
        x_i = self.encode(x)
        return self.features(x_i)

# linear feed-foward model with sigmoid activation functions
class LinNet(nn.Module):  
    def __init__(self, D_in, H1, H2, H3, D_out, xmin, xmax):
        super(LinNet, self).__init__()

        self.xmin = xmin
        self.xmax = xmax

        self.lin1 = nn.Linear(D_in, H1)
        self.lin2 = nn.Linear(H1,H1)
        self.lin3 = nn.Linear(H1,H2)
        self.lin4 = nn.Linear(H2,H2)
        self.lin5 = nn.Linear(H2,H3)
        self.lin6 = nn.Linear(H3, D_out)

    def forward(self, x):
        x_i = self.encode(x)
        out1 = torch.sigmoid(self.lin1(x_i))
        out2 = torch.sigmoid(self.lin2(out1))
        out3 = torch.sigmoid(self.lin3(out2))
        out4 = torch.sigmoid(self.lin4(out3))
        out5 = torch.sigmoid(self.lin5(out4))
        y_i = self.lin6(out5)
        return y_i     

    def encode(self,x):
        # convert x into numpy to do math
        x_np = x.data.cpu().numpy()
        xout = (x_np-self.xmin)/(self.xmax-self.xmin) - 0.5
        return Variable(torch.from_numpy(xout).type(dtype))

# linear feed-foward model with sigmoid activation functions
class MLP(nn.Module):  
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(MLP, self).__init__()

        # self.normfactor = normfactor

        self.mlp = nn.Sequential(OrderedDict([
            ('lin1',nn.Linear(D_in, H1)),
            # ('af1',nn.Sigmoid()),
            ('af1',nn.GELU()),
            ('lin2',nn.Linear(H1,H1)),
            # ('af2',nn.Sigmoid()),
            ('af2',nn.GELU()),
            ('lin3',nn.Linear(H1,H2)),
            # ('af3',nn.Sigmoid()),
            ('af3',nn.GELU()),
            ('lin4',nn.Linear(H2,H2)),
            # ('af4',nn.Sigmoid()),
            ('af4',nn.GELU()),
            ('lin5',nn.Linear(H2,H3)),
            # ('af5',nn.Sigmoid()),
            ('af5',nn.GELU()),
            ('lin6',nn.Linear(H3,D_out)), 
        ]))


    def forward(self, x):
#         x_i = self.encode(x)
        y_i = self.mlp(x)
        return y_i     

    # def encode(self,x):
    #     # convert x into numpy to do math
    #     x_np = x.data.cpu().numpy()
    #     xout = (x_np-self.xmin)/(self.xmax-self.xmin) - 0.5
    #     return Variable(torch.from_numpy(xout).type(dtype))

# Convolutional neural network (two convolutional layers)
class CNN(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1) ),
            ('af1', nn.GELU()),
            ('conv2', nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1) ),
            ('af2', nn.GELU()), 
        ]))
        
        self.mlp = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(32 * 5, 16) ),
            ('af3', nn.GELU()),
            ('lin2', nn.Linear(16, 8)),
        ]))

    def forward(self, x):
        y_i = self.cnn(x)
        y_i = y_i.view(y_i.size(0), -1)
        y = self.mlp(y_i)
        return y        
        