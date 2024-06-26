# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:38:34 2023

@author: yuxin dai
@e-mail:yuxindai@whu.edu.cn
"""

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import os
import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
import torch
from torch import nn, einsum
import torch.nn.functional as F

from pathlib import Path
from torch.optim import Adam
import sys
from numpy import shape
sys.path.append('..')
import numpy as np
import csv

#from torchvision.utils import save_image

print(torch.cuda.is_available)

# 判断变量是否为None
def exists(x):
    return x is not None

# 实现一个三元判断，传入val和default，如果val纯在，直接返回val，否则返回d
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 声明了一个残差网络
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# 上采样
def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

# 下采样
def Downsample(dim):
    return nn.Conv2d(5,5,(5,5),padding=1)


# 时间嵌入
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        
        
        half_dim_sin = self.dim // 2                
        embeddings_sin = math.log(10000) / (half_dim_sin - 1)
        embeddings_sin = torch.exp(torch.arange(half_dim_sin, device=device) * -embeddings_sin)
        embeddings_sin = time[:, None] * embeddings_sin[None, :]
        
        half_dim_cos = self.dim // 2 +1               
        embeddings_cos = math.log(10000) / (half_dim_cos - 1)
        embeddings_cos = torch.exp(torch.arange(half_dim_cos, device=device) * -embeddings_cos)
        embeddings_cos = time[:, None] * embeddings_cos[None, :]        
        
        embeddings = torch.cat((embeddings_sin.sin(), embeddings_cos.cos()), dim=-1)
        #print(r'embeddings.shape',embeddings.shape)
        return embeddings
    
# 基础的神经网络会用到的层，定义了层里面的两个基本操作，卷积和归一化以及激活函数
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
# 残差网络的构建
class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)




    
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Unet(nn.Module):
    def __init__(
        self,
        dim,

        with_time_emb=True,

    ):
        super().__init__()

        self.init_conv = nn.Conv2d(1, 4, 3, padding=1)

        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(20, 4))
        )
        self.res_conv=Residual(nn.Conv2d(4,4,(3,3),padding=1))
        self.attn= Residual(PreNorm(4, LinearAttention(4)))
        self.downsample=nn.Conv2d(4,8,(5,2),padding=1)
        self.upsample=nn.Conv2d(8,1,(1,4),padding=1)
        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

    def forward(self, x, time):
        x = self.init_conv(x)
        t = self.time_mlp(time) 
        t=self.mlp(t)      
        x = x + rearrange(t, "b c -> b c 1 1")

        x=self.res_conv(x)
        x=self.attn(x)
        x =self.downsample(x)
        x=self.upsample(x)
        #print(r'x',x.shape)
        return x

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    #rint(r'x_start',x_start.shape)
    if noise is None:
        noise = torch.randn_like(x_start)
    
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    
    predicted_noise = denoise_model(x_noisy, t)
    #print(r'noise',noise.shape,predicted_noise.shape,x_noisy.shape)    
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        #print(r'noise',noise.shape,predicted_noise.shape,x_noisy.shape,t.shape)
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# 这个函数就是使用模型，带入图片，不断去做采样
# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

# 函数入口
@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


timesteps = 200
# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# use seed for reproducability
torch.manual_seed(0)
image_size = 5
channels = 1


results_folder = Path(r"C:\Users\Desktop1\Desktop\边界样本生成\code")
# results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000


device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=image_size,
)
model.to(device)#加载到指定设备

optimizer = Adam(model.parameters(), lr=1e-3)

#load data
path=r'C:\Users\Desktop1\Desktop\边界样本生成\data'
data=np.load(path+r'\data_minmax.npy')
data=torch.tensor(data, dtype=torch.float32)


epochs = 20
batch_size = 128
for epoch in range(epochs):
    step = 0
    # for step, batch in enumerate(dataloader):
    for start, end in zip(
        range(0, len(data), batch_size),
        range(batch_size, len(data), batch_size)
        ):
      step = step+1
      optimizer.zero_grad()
      batch = data[start:end].reshape([-1, 1, 5, 2]).to(device)
      # batch_size = batch["pixel_values"].shape[0]
      # batch = batch["pixel_values"].to(device)      
      # 国内版启用这段，注释上面两行
      # batch_size = batch[0].shape[0]
      # batch = batch[0].to(device)
      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, timesteps, (batch_size,), device=device).long()
      
      loss = p_losses(model, batch, t, loss_type="huber")

      if step % 10 == 0:  # 本来是100
        print("Loss:", loss.item())

      loss.backward()
      optimizer.step()