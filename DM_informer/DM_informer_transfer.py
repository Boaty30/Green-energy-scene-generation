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

    # print(f'Noise shape: {noise.shape}, Predicted noise shape: {predicted_noise.shape}')  # 添加调试信息

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

# torch.manual_seed(0)
channels = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义Informer模型
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

# 加载预训练模型权重
pretrained_model_path = 'informer_ddpm_model.pth'
model.load_state_dict(torch.load(pretrained_model_path))
print("Pretrained model loaded.")

# 冻结预训练模型的部分层
for name, param in model.named_parameters():
    if 'encoder' in name:
        param.requires_grad = False

# 加载迁移学习的数据集
print("Loading transfer learning data...")
data1 = np.load("green_data1_15min.npy")
data2 = np.load("green_data2_15min.npy")

# 数据归一化
data1 = torch.tensor(data1, dtype=torch.float32)
data1_min = data1.min()
data1_max = data1.max()
data1 = (data1 - data1_min) / (data1_max - data1_min)

data2 = torch.tensor(data2, dtype=torch.float32)
data2_min = data2.min()
data2_max = data2.max()
data2 = (data2 - data2_min) / (data2_max - data2_min)

# 合并数据集
data = torch.cat((data1, data2), dim=0)
data = data.reshape(-1, 96, 1)
print("Transfer learning data shape:", data.shape)

epochs = 100
batch_size = 256

print("Starting transfer learning...")
loss_values = []

for epoch in range(epochs):
    step = 0
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        optimizer.zero_grad()
        
        batch = data[start:end].to(device)
        t = torch.randint(0, timesteps, (batch.shape[0],), device=device).long()
        
        loss = p_losses(model, batch, t, noise=None, loss_type="huber")
        
        if step % 100 == 0:
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item()}")
        
        loss.backward()
        optimizer.step()
        
        loss_values.append(loss.item())
        step += 1

plt.plot(loss_values)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Transfer Learning Loss over Time')
plt.savefig('transfer_learning_loss.png')


print("Transfer learning completed.")

# 保存迁移学习后的模型
torch.save(model.state_dict(), 'informer_ddpm_transfer_model.pth')
print("Transfer learning model saved.")
