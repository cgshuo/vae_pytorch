# -*- coding:utf-8 -*-
# @Time: 2020/11/30 6:53 下午
# @Author: cgshuo
# @Email: cgshuo@163.com
# @File: vae_pytorch.py

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torch.utils.data as Data
import torch.optim as optim
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
import torchsummary
from torchsummary import summary
#summary(your_model, input_size=(channels, H, W))

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

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
    def __init__(self, D_in, H, D_out): # input_dim, hidden_dim, hidden_dim
        super().__init__()
        self.liner1 = torch.nn.Linear(D_in, H) # dimension(D_in, H)
        self.liner2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.liner1(x))
        return F.relu(self.liner2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out): # latent_dim, hidden_dim, input_dim
        super().__init__()
        self.liner1 = torch.nn.Linear(D_in, H)
        self.liner2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.liner1(x))
        return F.relu(self.liner2(x))


class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder, latent_dim, hidden_dim): # 定义构造方法
        super().__init__() #调用父类方法
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(hidden_dim, latent_dim)  # 神经网络线性模块
        self._enc_log_sigma = torch.nn.Linear(hidden_dim, latent_dim)

    def _sample_latent(self, h_enc):
        """
        :param h_enc:
        :return: the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)  # y = e^x
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float() # 讲numpy转成tensor

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparametization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


if __name__ == '__main__':

    latent_dim = 8
    hidden_dim = 100
    input_dim = 28 * 28
    batch_size = 32

    transform = transforms.Compose([transforms.ToTensor()])  # pytorch 的一个图形库
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)
    dataloader = Data.DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=2)  # 用来把训练数据分成多个小组，此函数每次抛出一组数据。直至把所有的数据都抛出

    encoder = Encoder(input_dim, hidden_dim, hidden_dim)
    decoder = Decoder(latent_dim, hidden_dim, input_dim)
    vae = VAE(encoder, decoder, latent_dim, hidden_dim)

    criterion = nn.MSELoss()  # 均方损失函数： loss(x_i,y_i) = (x_i, y_i)^2

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)  # 为了使用torch.optim，需先构造一个优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数,lr学习率
    # Adam 自适应 SGD随机梯度下降
    l = 0
    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):
            inputs, classes = data
            inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes)
            optimizer.zero_grad()  # 梯度置0
            dec = vae(inputs)
            la_loss = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + la_loss
            loss.backward()  # 反向传播，计算当前梯度；
            optimizer.step()  # 根据梯度更新网络参数
            l += loss.item()
        print(epoch, l)

    plt.imshow(vae(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
    plt.show(block=True)





