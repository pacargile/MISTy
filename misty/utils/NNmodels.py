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

# ResNet convolutional neural network (two convolutional layers)
class ResNet(nn.Module):
    def __init__(self, D_in, H1, H2, D_out, xmin, xmax):
        super(ResNet, self).__init__()

        self.xmin = xmin
        self.xmax = xmax

        self.D_in = D_in
        self.H1    = H1
        self.H2    = H2
        self.D_out = D_out

        kernel_size = 1

        # self.features = nn.Sequential(
        #     nn.Linear(self.D_in, 32),
        #     nn.ReLU(),
        #     nn.Conv1d(1, 32, kernel_size, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 32, kernel_size, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 32, kernel_size, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 32, kernel_size, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Linear(32, self.H2),
        #     nn.ReLU(),
        #     nn.Linear(self.H2, self.D_out),
        # )

        # self.lstm = nn.LSTM(1, 32, 2, batch_first=True)
        # self.gru = nn.GRU(1, 32, 2, batch_first=True)
        # self.fc = nn.Linear(32, 1)
        
        # self.fc0 = nn.Sequential(
        #     nn.Linear(self.D_in, 32),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True)            
        # )

        self.conv1 = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=1),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16,16,kernel_size=1),
            nn.LeakyReLU(),
        )
        
        self.pool1 =nn.MaxPool1d(2)
        
        self.pool2 = nn.MaxPool1d(2)
        
        self.fc1 = nn.Sequential(
            nn.Linear(16,self.D_in),
            nn.LeakyReLU(),
            nn.Linear(self.D_in,self.D_out),
        )
        
        # self.deconv1 = nn.ConvTranspose1d(self.D_in, 64, kernel_size, stride=3, padding=5)
        # self.deconv2 = nn.ConvTranspose1d(64, 64, kernel_size, stride=3, padding=5)
        # self.deconv3 = nn.ConvTranspose1d(64, 64, kernel_size, stride=3, padding=5)
        # self.deconv4 = nn.ConvTranspose1d(64, 64, kernel_size, stride=3, padding=5)
        # self.deconv5 = nn.ConvTranspose1d(64, 64, kernel_size, stride=3, padding=5)
        # self.deconv6 = nn.ConvTranspose1d(64, 32, kernel_size, stride=3, padding=5)
        # self.deconv7 = nn.ConvTranspose1d(32, 1,  kernel_size, stride=3, padding=5)

        # self.deconv2b = nn.ConvTranspose1d(64, 64, 1, stride=3)
        # self.deconv3b = nn.ConvTranspose1d(64, 64, 1, stride=3)
        # self.deconv4b = nn.ConvTranspose1d(64, 64, 1, stride=3)
        # self.deconv5b = nn.ConvTranspose1d(64, 64, 1, stride=3)
        # self.deconv6b = nn.ConvTranspose1d(64, 32, 1, stride=3)

        # self.relu2 = nn.LeakyReLU()
        # self.relu3 = nn.LeakyReLU()
        # self.relu4 = nn.LeakyReLU()
        # self.relu5 = nn.LeakyReLU()
        # self.relu6 = nn.LeakyReLU()

        # self.dropout1 = nn.Dropout(p=0.2)
        # self.dropout2 = nn.Dropout(p=0.5)


    def forward(self, x):
        x_i = self.encode(x)

        x_i = x_i.unsqueeze(1)

        x_i = self.conv1(x_i)     
        x_i = self.pool1(x_i)
        x_i = self.conv2(x_i)
        x_i = self.pool2(x_i)
        x_i = x_i.view(x_i.shape[0],16)
        y_i = self.fc1(x_i)

        # x_i = x_i.unsqueeze(1)                
        # h0 = torch.zeros(2, x_i.size(0), 32).requires_grad_()
        # c0 = torch.zeros(2, x_i.size(0), 32).requires_grad_()
        # lstm_out, (hn, cn) = self.lstm(x_i, (h0.detach(), c0.detach()))
        # y_i = self.fc(lstm_out[:, -1, :]) 

        # x_i = self.fc0(x_i)

        # x_i = x_i.unsqueeze(2)                
        # print(x_i.shape)   
        # y_i = self.fc1(x_i)
        
        # x_i = x_i.view(x_i.shape[0], self.D_in)
        # y_i = self.features(x_i)#[:,None,:]

        return y_i


    def encode(self,x):
        # convert x into numpy to do math
        x_np = x.data.cpu().numpy()
        xout = (x_np-self.xmin)/(self.xmax-self.xmin) - 0.5
        return Variable(torch.from_numpy(xout).type(dtype))
