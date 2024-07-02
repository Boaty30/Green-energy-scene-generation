import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
font_path = 'C:\\Windows\\Fonts\\simsun.ttc'
font_prop = fm.FontProperties(fname=font_path, size=10)

# 加载生成样本和真实样本
generated_samples = np.load('DM_informer\sample_ddpm\ddpm_sample.npy')
real_samples = np.load('DM_informer\ca_data_99solar_15min.npy')
real_samples = real_samples.reshape(-1, 96, 1)

# 计算自相关系数
def calculate_acf(data, k):
    n = len(data)
    P = np.mean(data)
    sigma = np.std(data)
    acf = np.correlate(data - P, data - P, mode='full') / (sigma ** 2 * n)
    return acf[n-1:n+k-1]

selected_generated_samples = generated_samples[0:5]
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
    axs[0, i].plot(selected_generated_samples[i].squeeze(), label='生成样本')
    
    # 第二行：对应的真实样本
    axs[1, i].plot(selected_real_samples[i], color='red', label='最接近的真实样本')
    
    # 第三行：自相关系数
    gen_acf, real_acf = selected_acfs[i]
    axs[2, i].plot(gen_acf, label='生成样本的自相关系数', color='blue')
    axs[2, i].plot(real_acf, label='真实样本的自相关系数', linestyle='--', color='red')

# # 设置每一行的标题
# for i in range(5):
#     axs[0, i].set_title('生成样本', fontproperties=font_prop)
#     axs[1, i].set_title('最接近的真实样本', fontproperties=font_prop)
#     axs[2, i].set_title('自相关系数对比', fontproperties=font_prop)

# 设置每一行的标题
axs[0, 0].set_ylabel('生成样本', fontproperties=font_prop)
axs[1, 0].set_ylabel('最接近的真实样本', fontproperties=font_prop)
axs[2, 0].set_ylabel('自相关系数对比', fontproperties=font_prop)

# 在图外设置图例
lines_labels = [axs[2, i].get_legend_handles_labels() for i in range(5)]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[:2], labels[:2], loc='lower center', prop=font_prop, ncol=2)

# 设置总体标题和布局
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('DM_informer\sample_comparison_acf.png')
plt.show()
