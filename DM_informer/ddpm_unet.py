import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# 检查CUDA是否可用
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义一些辅助函数
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, device=device)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t).float()
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

# 定义SinusoidalPositionEmbeddings类
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# 定义Residual类
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# 定义PreNorm类
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# 定义LinearAttention类
class LinearAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.softmax(dim=-1), qkv)
        q = q * self.scale
        context = torch.einsum('bnd,bne->bde', q, k)
        out = torch.einsum('bde,bne->bnd', context, v)
        return self.to_out(out)

# 定义改进的UNet模型
class UNet(nn.Module):
    def __init__(self, dim, with_time_emb=True):
        super().__init__()
        self.init_conv = nn.Conv1d(1, 4, 3, padding=1)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(20, 4))
        self.res_conv = Residual(nn.Conv1d(4, 4, 3, padding=1))
        self.attn = Residual(PreNorm(4, LinearAttention(4)))
        self.downsample = nn.Conv1d(4, 8, 5, 2, padding=1)
        self.upsample = nn.Conv1d(8, 1, 1, 4, padding=1)
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
        t = self.mlp(t)
        x = x + t[:, :, None]
        x = self.res_conv(x)
        x = self.attn(x)
        x = self.downsample(x)
        x = self.upsample(x)
        return x

# 定义DDPM的参数
timesteps = 1000
betas = linear_beta_schedule(timesteps=timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# 加载数据
print("Loading data...")
data = np.load("ca_data_99solar_15min.npy")
data = torch.tensor(data, dtype=torch.float32).to(device)
data = data.reshape(-1, 1, 96)  # 调整数据形状以匹配UNet的输入
print("Data shape:", data.shape)

# 数据归一化
data_min = data.min().to(device)
data_max = data.max().to(device)
data = (data - data_min) / (data_max - data_min) * 2 - 1

# 定义模型和优化器
model = UNet(dim=20).to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

# 训练模型
epochs = 100
batch_size = 256
print("Starting training...")
loss_values = []

for epoch in range(epochs):
    step = 0
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        optimizer.zero_grad()

        batch = data[start:end]
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
plt.title('Training Loss over Time')
plt.savefig('ddpm_unet_loss.png')
plt.show()

# 训练完成后保存模型
torch.save(model.state_dict(), 'ddpm_unet_model.pth')
print("Model saved.")

# 生成样本的函数
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

@torch.no_grad()
def p_sample_loop(model, shape):
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
    generated_sample = p_sample_loop(model, shape=(1, 1, 96)).squeeze()

    # 将生成的样本转换为 tensor 并进行反归一化处理
    generated_sample_tensor = torch.tensor(generated_sample, dtype=torch.float32).to(device).squeeze()
    generated_sample_tensor = generated_sample_tensor * (data_max - data_min) / 2 + (data_max + data_min) / 2

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
    plt.savefig(f'sample_ddpm_unet/result{g}.png')
    plt.close()
    print(f'Sample {g+1} generated and saved.')
    generated_samples.append(generated_sample_tensor.cpu().numpy())

# 保存生成的样本到 .npy 文件
np.save('sample_ddpm_unet/ddpm_unet_sample.npy', generated_samples)
