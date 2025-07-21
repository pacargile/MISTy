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
            ('lin5',nn.Linear(H2,H2)),
            # ('af5',nn.Sigmoid()),
            ('af5',nn.GELU()),
            ('lin6',nn.Linear(H2,H2)),
            # ('af6',nn.Sigmoid()),
            ('af6',nn.GELU()),
            ('lin7',nn.Linear(H2,H2)),
            # ('af7',nn.Sigmoid()),
            ('af7',nn.GELU()),
            ('lin8',nn.Linear(H2,H3)),
            # ('af8',nn.Sigmoid()),
            ('af8',nn.GELU()),
            ('lin9',nn.Linear(H3,D_out)), 
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
# class CNN(nn.Module):
#     def __init__(self, D_in, H1, H2, H3, D_out):
#         super(CNN, self).__init__()

#         self.D_in = D_in
#         self.H1 = H1
#         self.H2 = H2
#         self.H3 = H2
#         self.D_out = D_out

#         self.cnn = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1) ),
#             ('af1', nn.GELU()),
#             ('conv2', nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1) ),
#             ('af2', nn.GELU()), 
#         ]))
        
#         self.mlp = nn.Sequential(OrderedDict([
#             ('lin1', nn.Linear(32*self.D_in, self.H1) ),
#             ('af3', nn.GELU()),
#             ('lin2', nn.Linear(self.H1, self.H2)),
#             ('af4', nn.GELU()),
#             ('lin2', nn.Linear(self.H2, self.D_out)),
#         ]))

#     def forward(self, x):
#         # x = x.view(x.size(0),1,x.size(-1))
#         x = x.unsqueeze(1)
#         y_i = self.cnn(x)
#         y_i = y_i.view(x.size(0),-1)
#         y = self.mlp(y_i)
#         return y        
# Convolutional neural network (two convolutional layers)
class CNN(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(CNN, self).__init__()

        self.D_in = D_in
        self.H1 = H1
        self.H2 = H2
        self.D_out = D_out

        # CNN Layers
        self.cnn = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)),  # Shape: (batch_size, 16, 5)
            ('af1', nn.GELU()),
            ('conv2', nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)), # Shape: (batch_size, 32, 5)
            ('af2', nn.GELU()),
            ('conv3', nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)), # Shape: (batch_size, 32, 5)
            ('af3', nn.GELU()),
        ]))

        # MLP Layers
        self.mlp = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(64 * D_in, H1)),  # Fully connected input matches flattened size
            ('af4', nn.GELU()),
            ('lin2', nn.Linear(H1, H2)),     # Hidden layer
            ('af5', nn.GELU()),
            ('lin3', nn.Linear(H2, D_out)), # Output layer
        ]))

    def forward(self, x):
        # Reshape input to add channel dimension
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, 5)

        # Pass through CNN
        y_i = self.cnn(x)  # Shape: (batch_size, 32, 5)

        # Flatten for MLP
        y_i = y_i.view(x.size(0), -1)  # Shape: (batch_size, 32 * 5)

        # Pass through MLP
        y = self.mlp(y_i)  # Shape: (batch_size, D_out)
        return y
    

# New Approach using TrackGenerator and AgeSelector

# ------------------------------
# Track Generator Module
# ------------------------------

class TrackGenerator(nn.Module):
    def __init__(self, input_dim=3, latent_dim=32, latent_steps=128, hidden_dim=256):
        """
        input_dim: (Mi, [Fe/H], [alpha/Fe])
        latent_dim: dimensionality of latent representation per step
        latent_steps: number of steps in latent trajectory (e.g., 128)
        hidden_dim: hidden layer width
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Output latent trajectory: (latent_steps, latent_dim)
        self.decoder = nn.Linear(hidden_dim, latent_steps * latent_dim)
        self.latent_dim = latent_dim
        self.latent_steps = latent_steps

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        returns latent trajectory: (batch_size, latent_steps, latent_dim)
        """
        x = self.encoder(x)  # (batch_size, hidden_dim)
        latent = self.decoder(x)  # (batch_size, latent_steps * latent_dim)
        latent = latent.view(-1, self.latent_steps, self.latent_dim)
        return latent


# ------------------------------
# Age Selector Module
# ------------------------------

class AgeSelector(nn.Module):
    def __init__(self, latent_dim=32, latent_steps=128, hidden_dim=128, output_dim=6):
        """
        latent_dim: same as TrackGenerator latent_dim
        latent_steps: number of latent steps
        output_dim: number of predicted physical parameters (e.g., Teff, logg, etc.)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Simple MLP selector: (age + pooled latent) → output labels
        self.selector = nn.Sequential(
            nn.Linear(1 + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, age, latent):
        """
        age: (batch_size, 1)
        latent: (batch_size, latent_steps, latent_dim)
        """
        # Soft pooling: compute soft attention weights from age
        batch_size = latent.size(0)
        steps = torch.linspace(0, 1, latent.size(1), device=age.device).unsqueeze(0)  # (1, latent_steps)
        
        # Normalize age into [0, 1] domain for attention
        age_norm = age / age.max()
        attn_weights = torch.exp(-100 * (steps - age_norm).pow(2))  # Gaussian-like attention
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True)  # normalize
        
        # Weighted sum over latent steps
        pooled_latent = torch.sum(latent * attn_weights.unsqueeze(-1), dim=1)  # (batch_size, latent_dim)
        
        # Concatenate age input
        selector_input = torch.cat([age, pooled_latent], dim=1)  # (batch_size, 1 + latent_dim)
        
        return self.selector(selector_input)


# ------------------------------
# Combined Model
# ------------------------------

class TwoStep(nn.Module):
    def __init__(self, input_dim=4, output_dim=6, latent_dim=32, latent_steps=128):
        """
        input_dim: (Mi, [Fe/H], [alpha/Fe], age)
        output_dim: predicted physical labels
        """
        super().__init__()
        self.track_gen = TrackGenerator(input_dim=3, latent_dim=latent_dim, latent_steps=latent_steps)
        self.age_selector = AgeSelector(latent_dim=latent_dim, latent_steps=latent_steps, output_dim=output_dim)

    def forward(self, x):
        """
        x: (batch_size, 4) → (Mi, [Fe/H], [alpha/Fe], age)
        """
        track_input = x[:, :3]  # (Mi, [Fe/H], [alpha/Fe])
        age = x[:, 3:4]  # age
        
        latent = self.track_gen(track_input)
        output = self.age_selector(age, latent)
        return output