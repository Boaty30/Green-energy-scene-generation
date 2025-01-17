import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.font_manager import FontProperties

# 设置中文字体
font_path = 'C:\\Windows\\Fonts\\simsun.ttc'
font_prop = FontProperties(fname=font_path, size=10)

# 加载数据
generated_samples = np.load('GAN&VAE\sample_wgan\generated_samples.npy').flatten()
real_samples = np.load('DM_informer\ca_data_99solar_15min.npy').flatten()

# 计算CDF
def calculate_cdf(data):
    data_sorted = np.sort(data)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    return data_sorted, cdf

gen_data_sorted, gen_cdf = calculate_cdf(generated_samples)
real_data_sorted, real_cdf = calculate_cdf(real_samples)

# 计算PDF
def calculate_pdf(data, bins=100):
    pdf, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, pdf

gen_bin_centers, gen_pdf = calculate_pdf(generated_samples)
real_bin_centers, real_pdf = calculate_pdf(real_samples)

# 绘制CDF对比图
plt.figure(figsize=(6, 6))
plt.plot(real_data_sorted, real_cdf, label='真实数据', linestyle='--', color='red')
plt.plot(gen_data_sorted, gen_cdf, label='生成数据', color='blue')
plt.xlabel('值', fontproperties=font_prop)
plt.ylabel('CDF', fontproperties=font_prop)
plt.title('CDF对比', fontproperties=font_prop)
plt.legend(prop=font_prop)

# 保存CDF图像
plt.savefig('GAN&VAE\cdf_comparison.png')
plt.show()

# 绘制PDF对比图
plt.figure(figsize=(6, 6))
plt.plot(real_bin_centers, real_pdf, label='真实数据', linestyle='--', color='red')
plt.plot(gen_bin_centers, gen_pdf, label='生成数据', color='blue')
plt.xlabel('值', fontproperties=font_prop)
plt.ylabel('PDF', fontproperties=font_prop)
plt.title('PDF对比', fontproperties=font_prop)
plt.legend(prop=font_prop)

# 保存PDF图像
plt.savefig('GAN&VAE\pdf_comparison.png')
plt.show()
