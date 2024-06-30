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

print("Loading transfer learning data...")
data1 = np.load("DM_informer/green_data1_15min.npy")
data2 = np.load("DM_informer/green_data2_15min.npy")

# 数据归一化
data1 = torch.tensor(data1, dtype=torch.float32)
data1_min = data1.min()
data1_max = data1.max()
# data1 = (data1 - data1_min) / (data1_max - data1_min)

data2 = torch.tensor(data2, dtype=torch.float32)
data2_min = data2.min()
data2_max = data2.max()
# data2 = (data2 - data2_min) / (data2_max - data2_max)

# 合并数据集
data = torch.cat((data1, data2), dim=0)
data = data.reshape(-1, 96, 1)
data_max = data.max()
data_min = data.min()
print("Transfer learning data shape:", data.shape)

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

# forward diffusion
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

# 定义迁移学习后的Informer模型
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

# 加载迁移学习后的模型权重
model.load_state_dict(torch.load('DM_informer/informer_ddpm_transfer_model.pth'))
model.eval()
print("Transfer learning model loaded.")

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
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
    return img.cpu().numpy()  # 返回最后一步生成的样本

# 生成样本
batch_size = 256  # 设置批量大小
num_samples = 10000
generated_samples = []

for i in range(0, num_samples, batch_size):
    batch_generated_samples = p_sample_loop(model, shape=(batch_size, 96, 1))
    for g in range(batch_size):
        generated_sample = batch_generated_samples[g]
        generated_sample_tensor = torch.tensor(generated_sample, dtype=torch.float32).squeeze()
        generated_sample_tensor = generated_sample_tensor * (data_max - data_min) + data_min
        generated_sample_tensor = torch.clamp(generated_sample_tensor, min=0)
        generated_samples.append(generated_sample_tensor.cpu().numpy())

    print(f'Batch {i // batch_size + 1}/{num_samples // batch_size} generated and saved.')

generated_samples = np.array(generated_samples)

# 保存生成的样本到 .npy 文件
np.save('DM_informer/sample_transfer/ddpm_transfer_sample.npy', generated_samples)
print(f'{num_samples} samples generated and saved.')
