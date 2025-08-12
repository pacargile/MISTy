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
# Utility: Residual MLP Block
# ------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim, hidden=None, dropout=0.0):
        super().__init__()
        h = hidden or dim
        self.fc1 = nn.Linear(dim, h)
        self.fc2 = nn.Linear(h, dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.act(self.fc1(self.norm(x)))
        y = self.dropout(y)
        y = self.fc2(self.act(y))
        return x + y

# ------------------------------
# Phase warp for step positions
# alpha in [0,1]: larger -> more density at late phases
# ------------------------------
def phase_warp(u, alpha=0.6):
    # convex warp; u in [0,1]
    return (1 - alpha) * u + alpha * u * u

# ------------------------------
# Track Generator (deeper, residual)
# ------------------------------
class TrackGenerator(nn.Module):
    def __init__(self, input_dim=3, latent_dim=128, latent_steps=512,
                 hidden_dim=512, nblocks=3, dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_steps = latent_steps

        self.inp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout=dropout) for _ in range(nblocks)])
        self.dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_steps * latent_dim),
        )

    def forward(self, x):
        h = self.inp(x)
        h = self.blocks(h)
        lat = self.dec(h)
        return lat.view(-1, self.latent_steps, self.latent_dim)

# ------------------------------
# Age-Selector with learned center and temperature
# Optional conditioning on (Mi, FeH, aFe)
# ------------------------------
class AgeSelectorAttn(nn.Module):
    def __init__(self, latent_dim=128, latent_steps=512, hidden_dim=512, output_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_steps = latent_steps

        # control: predicts attention center s in [0,1] and temperature tau>0
        # inputs: age (1) + cond (3) = 4
        self.ctrl = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * output_dim)  # [mu, log_sigma] per output
        )

        # precompute warped step positions
        steps = torch.linspace(0, 1, latent_steps)
        self.register_buffer("steps", phase_warp(steps)[None, :])  # (1, S)

    def forward(self, age, latent, cond):  # cond = (Mi, FeH, aFe)
        # ctrl
        ctrl_in = torch.cat([age, cond], dim=-1)    # (B, 4)
        s, log_tau = self.ctrl(ctrl_in).chunk(2, dim=-1)
        s = torch.sigmoid(s)                        # center in [0,1]
        tau = F.softplus(log_tau) + 1e-3            # width > 0

        # attention over warped steps
        d2 = (self.steps - s).pow(2)                # (B, S)
        attn = torch.softmax(-d2 / tau.clamp_min(1e-3), dim=1)  # (B, S)

        pooled = torch.sum(latent * attn.unsqueeze(-1), dim=1)  # (B, D)

        # heteroscedastic head
        out = self.head(pooled)                     # (B, 2*D_out)
        mu, log_sigma = out.chunk(2, dim=-1)        # each (B, D_out)
        return mu, log_sigma

# ------------------------------
# Combined Model (Upgraded)
# ------------------------------
class TwoStep(nn.Module):
    def __init__(self, input_dim=4, output_dim=8,
                 latent_dim=128, latent_steps=512, hidden_dim=512,
                 tg_blocks=3, dropout=0.0):
        super().__init__()
        self.output_dim = output_dim
        self.track_gen = TrackGenerator(
            input_dim=3, latent_dim=latent_dim, latent_steps=latent_steps,
            hidden_dim=hidden_dim, nblocks=tg_blocks, dropout=dropout
        )
        self.age_selector = AgeSelectorAttn(
            latent_dim=latent_dim, latent_steps=latent_steps,
            hidden_dim=hidden_dim, output_dim=output_dim
        )

    def forward(self, x, return_variance=False):
        # x: (B, 4) = (Mi, FeH, aFe, log_age)
        cond = x[:, :3]
        age  = x[:, 3:4]
        latent = self.track_gen(cond)
        mu, log_sigma = self.age_selector(age, latent, cond)
        return (mu, log_sigma) if return_variance else mu
    
# ------------------------------
# Normalization Wrapper
# ------------------------------


class NormalizedModel(nn.Module):
    def __init__(self, model, input_mean, input_std, output_mean, output_std):
        super().__init__()
        self.model = model
        self.register_buffer("input_mean", torch.tensor(input_mean).float())
        self.register_buffer("input_std", torch.tensor(input_std).float())
        self.register_buffer("output_mean", torch.tensor(output_mean).float())
        self.register_buffer("output_std", torch.tensor(output_std).float())

    def normalize_input(self, x):
        return (x - self.input_mean) / self.input_std

    def denormalize_output(self, y):
        return y * self.output_std + self.output_mean

    def forward(self, x):
        x_norm = self.normalize_input(x)
        y_norm = self.model(x_norm)
        return self.denormalize_output(y_norm)