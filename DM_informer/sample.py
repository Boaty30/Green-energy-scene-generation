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
data = np.load("ca_data_99solar_15min.npy")
data = torch.tensor(data, dtype=torch.float32)

# 数据归一化
data_min = data.min()
data_max = data.max()
# data = (data - data_min) / (data_max - data_min)

data = data.reshape(-1, 96, 1)
print("Data shape:", data.shape)  # 输出数据形状

# 加载模型权重
model.load_state_dict(torch.load('informer_ddpm_model.pth'))
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
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs[-1]  # 返回最后一步生成的样本

generated_samples = []
for g in range(30):
    # 生成样本
    generated_sample = p_sample_loop(model, shape=(1, 96, 1)).squeeze()

    # 将生成的样本转换为 tensor 并进行反归一化处理
    generated_sample_tensor = torch.tensor(generated_sample, dtype=torch.float32).squeeze()
    generated_sample_tensor = generated_sample_tensor * (data_max - data_min) + data_min

    # 确保样本中所有的值不小于0
    generated_sample_tensor = torch.clamp(generated_sample_tensor, min=0)

    # 计算欧几里得距离
    def euclidean_distance(tensor1, tensor2):
        return torch.sqrt(torch.sum((tensor1 - tensor2) ** 2))

    # 生成样本并计算与数据集中样本的距离
    distances = []
    for i in range(data.shape[0]):
        distances.append(euclidean_distance(generated_sample_tensor, data[i].squeeze()))

    distances = torch.tensor(distances)
    min_distance_index = torch.argmin(distances).item()

    # 找到距离最小的样本
    closest_sample = data[min_distance_index].cpu().numpy()

    # 可视化对比
    plt.figure(figsize=(12, 6))
    plt.plot(closest_sample, label='Closest Sample')
    plt.plot(generated_sample_tensor.cpu().numpy(), label='Generated Sample')
    plt.legend()
    plt.title('Generated Sample vs. Closest Sample in Dataset')
    plt.xlabel('Hour')
    plt.ylabel('Output')
    # plt.show()
    plt.savefig(f'sample_ddpm/result{g}.png')
    plt.close()
    print(f'Sample {g+1} generated and saved.')
    generated_samples.append(generated_sample_tensor.cpu().numpy())

# 保存生成的样本到 .npy 文件
np.save('sample_ddpm/ddpm_sample.npy', generated_samples)
