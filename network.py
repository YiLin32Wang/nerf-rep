from math import pi
import os
from turtle import forward
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF_net(nn.Module):
    def __init__(self,
                 Lx = 10,
                 Ld = 4,
                 in_x_channel=3,
                 in_d_channel=3,
                 out_sigma_channel=1,
                 out_c_channel=3,
                 sigma_fcs_num=8,
                 sigma_fcs_channel=256,
                 sigma_fcs_activate=nn.ReLU,
                 c_fcs_num=1,
                 c_fcs_channel=128,
                 c_fcs_activate=nn.ReLU,
                 activate_inplace=False):
        super().__init__()
        self.Lx = Lx
        self.Ld = Ld
        self.in_x_channel= in_x_channel * Lx
        self.in_d_channel= in_d_channel * Ld
        self.out_sigma_channel = out_sigma_channel
        self.out_c_channel = out_c_channel
        self.sigma_fcs_num=sigma_fcs_num
        self.sigma_fcs_channel=sigma_fcs_channel
        self.sigma_fcs_activate=sigma_fcs_activate(inplace=activate_inplace)
        self.c_fcs_num=c_fcs_num
        self.c_fcs_channel=c_fcs_channel
        self.c_fcs_activate=c_fcs_activate(inplace=activate_inplace)
        
        sigma_layers = [
            nn.Linear(self.in_x_channel, self.sigma_fcs_channel), 
            self.sigma_fcs_activate()
            ]
        for i in range(self.sigma_fcs_num - 1):
            sigma_layers.append[
            nn.Linear(self.sigma_fcs_channel, self.sigma_fcs_channel), 
            self.sigma_fcs_activate() # TODO: should ther be an activation function?
            ]
        sigma_layers.append[
            nn.Linear(self.sigma_fcs_channel, self.sigma_fcs_channel + self.out_sigma_channel), 
            self.sigma_fcs_activate()
        ] # the final output layer of volume density and feature vector
        self.sigma_fcs = nn.Sequential(*sigma_layers)
        
        c_layers = [
            nn.Linear(self.in_d_channel + self.sigma_fcs_channel, self.c_fcs_channel), 
            self.sigma_fcs_activate(),
            nn.Linear(self.c_fcs_channel, self.out_c_channel),
            self.c_fcs_activate()
            ]
        self.c_fcs = nn.Sequential(*c_layers)
        
    def gamma_mapping(self, x):
        B, C, N = x.shape
        for i in range(C):
            x_comp = x[:, i, :].reshape(B, 1, -1)
            gamma = []
            for i in range(self.Lx):
                theta = (2**i) * pi * x_comp
                gamma += [torch.sin(theta), torch.cos(theta)]
            gamma = torch.cat(gamma, dim=1)
            x_gamma.append(gamma)
        x_gamma = torch.cat(x_gamma, dim=1)
        return x_gamma
        
    def positional_encoding(self, x, d):
        B, Cc, Nx = x.shape
        B, Cd, Nd = d.shape
        
        x_gamma, d_gamma = [], []
        x_gamma = self.gamma_mapping(x)
        d_gamma = self.gamma_mapping(d)
        
        return x_gamma, d_gamma
        
        
    def forward(self, x, d):
        """_summary_

        Args:
            x (_type_): gamma mapped x
            d (_type_): gamma mapped d
        """
        bs = x.shape[0]
        
        x = self.sigma_fcs(x)
        sigma, fx = x[:, :self.out_sigma_channel, :], x[:, self.out_sigma_channel:, :]
        
        x_c = torch.cat([fx, d], dim=1)
        c = self.c_fcs(x_c)
        
        return sigma, c
        
        
        
        
        
        
        