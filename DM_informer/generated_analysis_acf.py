import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
font_path = 'C:\\Windows\\Fonts\\simsun.ttc'
font_prop = fm.FontProperties(fname=font_path, size=10)

# 加载生成样本和真实样本
#generated_samples = np.load(r'C:\Users\boaty\Documents\WHU Files\硕士项目\综合能源场景生成\GAN\sample_wgan\generated_samples.npy')
generated_samples = np.load('sample_ddpm\ddpm_sample.npy')
real_samples = np.load('ca_data_99solar_15min.npy')
real_samples = real_samples.reshape(-1, 96, 1)

# 计算自相关系数
def calculate_acf(data, k):
    n = len(data)
    P = np.mean(data)
    sigma = np.std(data)
    acf = np.correlate(data - P, data - P, mode='full') / (sigma ** 2 * n)
    return acf[n-1:n+k-1]

selected_generated_samples = generated_samples[:5]
selected_real_samples = []
selected_acfs = []

for gen_sample in selected_generated_samples:
    gen_sample = gen_sample.squeeze()
    distances = np.sqrt(np.sum((real_samples.squeeze() - gen_sample) ** 2, axis=1))
    closest_idx = np.argmin(distances)
    closest_sample = real_samples[closest_idx].squeeze()
    selected_real_samples.append(closest_sample)
    
    # 计算自相关系数
    gen_acf = calculate_acf(gen_sample, 20)
    real_acf = calculate_acf(closest_sample, 20)
    selected_acfs.append((gen_acf, real_acf))

# 绘制图表
fig, axs = plt.subplots(3, 5, figsize=(20, 12))

for i in range(5):
    # 第一行：生成样本
    axs[0, i].plot(selected_generated_samples[i].squeeze())
    
    # 第二行：对应的真实样本
    axs[1, i].plot(selected_real_samples[i], color='red')
    
    # 第三行：自相关系数
    gen_acf, real_acf = selected_acfs[i]
    axs[2, i].plot(gen_acf, label='生成样本的自相关系数')
    axs[2, i].plot(real_acf, label='真实样本的自相关系数', linestyle='--')
    axs[2, i].legend(prop=font_prop)

# 设置每一行的标题
axs[0, 0].set_ylabel('生成样本', fontproperties=font_prop)
axs[1, 0].set_ylabel('最接近的真实样本', fontproperties=font_prop)
axs[2, 0].set_ylabel('自相关系数对比', fontproperties=font_prop)

# 设置总体标题和布局
# plt.suptitle('生成样本、最接近的真实样本及其自相关系数对比', fontproperties=font_prop)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('sample_comparison_acf.png')
plt.show()
