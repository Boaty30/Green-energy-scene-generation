import numpy as np
import matplotlib.pyplot as plt

# 加载生成样本和真实样本
generated_samples = np.load('GAN\sample_vae\generated_samples.npy')
real_samples = np.load('DM_informer/ca_data_99solar_15min.npy')

# 将真实样本重新调整形状
real_samples = real_samples.reshape(-1, 96, 1)

# 计算功率区间平均宽度
def calculate_mean_width(samples):
    P_t_up = np.max(samples, axis=0)
    P_t_down = np.min(samples, axis=0)
    W = np.mean(P_t_up - P_t_down)
    return W

# 计算生成样本和真实样本的功率区间平均宽度
generated_mean_width = calculate_mean_width(generated_samples)
real_mean_width = calculate_mean_width(real_samples)

# 输出结果
print(f'生成样本的功率区间平均宽度: {generated_mean_width}')
print(f'真实样本的功率区间平均宽度: {real_mean_width}')

# 将结果保存到CSV文件
results = {
    '样本类型': ['生成样本', '真实样本'],
    '功率区间平均宽度': [generated_mean_width, real_mean_width]
}

import pandas as pd
df = pd.DataFrame(results)
df.to_csv('GAN\VAE\power_interval_mean_width.csv', index=False, encoding='utf_8_sig')

# 展示结果
print("功率区间平均宽度已保存到 'power_interval_mean_width.csv'")

# 获取真实数据和生成数据的数量
N, T, _ = real_samples.shape
M = generated_samples.shape[0]

# 初始化覆盖数量
coverage_count = 0

# 遍历每个时刻点
for t in range(T):
    # 获取生成数据在该时刻的最小值和最大值
    P_min = np.min(generated_samples[:, t])
    P_max = np.max(generated_samples[:, t])
    
    # 统计真实数据在该时刻是否在生成数据的范围内
    n_t = np.sum((real_samples[:, t, 0] >= P_min) & (real_samples[:, t, 0] <= P_max))
    coverage_count += n_t

# 计算覆盖率
coverage_rate = coverage_count / (N * T)
print(f"覆盖率: {coverage_rate}")

# 保存覆盖率结果到CSV文件
import csv

with open('GAN\VAE\coverage_rate.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['样本类型', '覆盖率'])
    writer.writerow(['生成样本', coverage_rate])

print("覆盖率结果已保存到coverage_rate.csv文件中。")
