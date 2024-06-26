import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from torch import nn, einsum
import torch.nn.functional as F

from pathlib import Path
from torch.optim import Adam
import numpy as np

# 导入Informer模型
from model import InformerForDDPM

print("CUDA available:", torch.cuda.is_available())

# 判断变量是否为None
def exists(x):
    return x is not None

# 实现一个三元判断，传入val和default，如果val存在，直接返回val，否则返回d
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

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
    if noise is None:
        noise = torch.randn_like(x_start)
    
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)
    
    # 确保输出形状匹配
    if predicted_noise.shape != noise.shape:
        predicted_noise = predicted_noise[:, :x_start.size(1), :x_start.size(2)]

    print(f'Noise shape: {noise.shape}, Predicted noise shape: {predicted_noise.shape}')  # 添加调试信息

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

timesteps = 200
betas = linear_beta_schedule(timesteps=timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

torch.manual_seed(0)
image_size = 5
channels = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义Informer模型
model = InformerForDDPM(
    enc_in=1, 
    seq_len=24,  
    d_model=512,
    n_heads=8,
    e_layers=3,
    d_ff=512,
    dropout=0.1,
    attn='prob',
    activation='gelu',
    output_attention=False,
    distil=True,
    device=device
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

# 加载数据
print("Loading data...")
data = np.load("C:\\Users\\boaty\\Documents\\WHU Files\\硕士项目\\综合能源场景生成\\Renewable Scenario Generation\\dataset\\ca_data_99solar.npy")
data = torch.tensor(data, dtype=torch.float32)
data = data[0]
data = data.reshape(365, 24, 1)
print("Data shape:", data.shape)  # 输出数据形状

epochs = 200
batch_size = 16

print("Starting training...")
for epoch in range(epochs):
    step = 0
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        optimizer.zero_grad()
        
        batch = data[start:end].to(device)
        t = torch.randint(0, timesteps, (batch.shape[0],), device=device).long()
        
        # 调试信息：输出每个批次的形状
        print(f"Epoch {epoch+1}, Step {step+1}: Batch shape: {batch.shape}")
        
        loss = p_losses(model, batch, t, noise=None, loss_type="huber")
        
        if step % 10 == 0:
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item()}")
        
        loss.backward()
        optimizer.step()
        
        step += 1

print("Training completed.")
