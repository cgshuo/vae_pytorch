# -*- coding:utf-8 -*-
# @Time: 2020/11/30 6:53 下午
# @Author: cgshuo
# @Email: cgshuo@163.com
# @File: vae_pytorch.py

import torch
import torch.nn.functional as F

class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim) # create a tensor
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.liner1 = torch.nn.Linear(D_in, H) # dimension(D_in, H)
        self.liner2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.liner1(x))
        return F.relu(self.liner2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.liner1 = torch.nn.Linear(D_in, H)
        self.liner2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.liner1(x))
        return F.relu(self.liner2(x))




