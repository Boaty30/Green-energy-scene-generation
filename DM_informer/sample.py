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

# Import the Informer model
from model import InformerForDDPM

print("CUDA available:", torch.cuda.is_available())

# Define some utility functions
def exists(x):
    return x is not None

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

# Forward diffusion
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
    
    if predicted_noise.shape != noise.shape:
        predicted_noise = predicted_noise[:, :x_start.size(1), :x_start.size(2)]

    print(f'Noise shape: {noise.shape}, Predicted noise shape: {predicted_noise.shape}')

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

timesteps = 1000
betas = linear_beta_schedule(timesteps=timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the Informer model
model = InformerForDDPM(
    enc_in=1, 
    seq_len=96,  
    d_model=512,  
    n_heads=8,  
    e_layers=4,  
    d_ff=2048,  
    dropout=0.2,  
    attn='prob',
    activation='gelu',
    output_attention=False,
    distil=True,
    device=device
)

model.to(device)

optimizer = Adam(model.parameters(), lr=1e-4)

# Load data
print("Loading data...")
data = np.load("DM_informer/ca_data_99solar_15min.npy")
data = torch.tensor(data, dtype=torch.float32)

# 数据归一化
data_min = data.min()
data_max = data.max()
# data = (data - data_min) / (data_max - data_min)

data = data.reshape(-1, 96, 1)
print("Data shape:", data.shape)  # 输出数据形状

# 加载模型权重
model.load_state_dict(torch.load('DM_informer/informer_ddpm_model.pth'))
model.eval()
print("Model loaded.")

# 定义 p_sample 函数
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# 定义 p_sample_loop 函数
@torch.no_grad()
def p_sample_loop(model, shape, batch_size=256):
    device = next(model.parameters()).device
    img = torch.randn((batch_size, *shape), device=device)
    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), i)
    return img.cpu().numpy()  # 返回最后一步生成的样本

# 批量生成样本
batch_size = 256  # 可以根据实际情况调整批量大小
num_samples = 10000
generated_samples = []

for idx in range(num_samples // batch_size):
    # 生成样本
    batch_generated_samples = p_sample_loop(model, shape=(96, 1), batch_size=batch_size)

    # 将生成的样本转换为 tensor 并进行反归一化处理
    batch_generated_samples = torch.tensor(batch_generated_samples, dtype=torch.float32)
    batch_generated_samples = batch_generated_samples * (data_max - data_min) + data_min

    # 确保样本中所有的值不小于0
    batch_generated_samples = torch.clamp(batch_generated_samples, min=0)

    generated_samples.append(batch_generated_samples.cpu().numpy())

    # 输出当前进度
    print(f'Batch {idx+1}/{num_samples // batch_size} generated and saved.')

# 将所有生成的样本合并成一个数组
generated_samples = np.concatenate(generated_samples, axis=0)

# 保存生成的样本到 .npy 文件
np.save('DM_informer/sample_ddpm/ddpm_sample.npy', generated_samples)
print(f'{num_samples} samples generated and saved.')
