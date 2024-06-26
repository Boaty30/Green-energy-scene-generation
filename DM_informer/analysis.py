import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 加载数据
data = np.load("ca_data_99solar_15min.npy")
data = data.reshape(-1, 96)  # 将数据重塑为 (num_samples, 96)

# 标准化数据
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# 使用 K-means 进行聚类
num_clusters = 10  # 可以根据需要调整聚类数量
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(data_normalized)

# 获取每个聚类的中心
centroids = kmeans.cluster_centers_

# 将中心反标准化
centroids_original = scaler.inverse_transform(centroids)

# 绘制每个典型曲线
plt.figure(figsize=(12, 8))
for i, centroid in enumerate(centroids_original):
    plt.plot(centroid, label=f'Cluster {i + 1}')

plt.legend()
plt.title('典型曲线')
plt.xlabel('Hour')
plt.ylabel('Output')
plt.show()
plt.savefig('typical_curves.png')
