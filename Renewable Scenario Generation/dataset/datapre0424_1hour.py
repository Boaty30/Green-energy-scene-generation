# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:12:19 2024

@author: Admin
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Folder path
folder_path = r'C:\Users\boaty\Documents\WHU Files\硕士项目\综合能源场景生成\Renewable Scenario Generation\dataset\ca-pv-2006'  # replace it with your folder path

# The pattern which will be used to match the file names
pattern = r"_([0-9.]+)_(-?[0-9.]+)_(\d{4})_(UPV|DPV)_"

# Lists to store the latitudes, longitudes and technology type(UPV/DPV)
latitudes = []
longitudes = []
tech_types = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        match = re.search(pattern, filename)
        if match:
            lat = float(match.group(1))  
            lon = float(match.group(2))  
            tech_type = match.group(4)  
            latitudes.append(lat)
            longitudes.append(lon)
            tech_types.append(tech_type)

print("Latitudes:", latitudes)
print("Longitudes:", longitudes)
print("Technology types:", tech_types)

plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.size'] = 16
latitudes_UPV = [latitudes[i] for i in range(len(tech_types)) if tech_types[i] == 'UPV']
longitudes_UPV = [longitudes[i] for i in range(len(tech_types)) if tech_types[i] == 'UPV']
latitudes_DPV = [latitudes[i] for i in range(len(tech_types)) if tech_types[i] == 'DPV']
longitudes_DPV = [longitudes[i] for i in range(len(tech_types)) if tech_types[i] == 'DPV']

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(latitudes_DPV, longitudes_DPV, color='#E73744', label='DPV', marker='^', s=10)

rectangle = patches.Rectangle((33.4, -119.4), 1.1, 2.6, linewidth=1, edgecolor='black', facecolor='none')
ax.add_patch(rectangle)

rectangle = patches.Rectangle((32.5, -117.5), 0.8, 0.8, linewidth=1, edgecolor='green', facecolor='none')
ax.add_patch(rectangle)

rectangle = patches.Rectangle((35.2, -119.5), 0.3, 0.8, linewidth=1, edgecolor='green', facecolor='none')
ax.add_patch(rectangle)

ax.set_xlabel('纬度')
ax.set_ylabel('经度')
ax.set_title('加利福尼亚州光伏场站地理分布')
plt.savefig(r'C:\Users\boaty\Documents\WHU Files\硕士项目\综合能源场景生成\Renewable Scenario Generation\dataset\ca-pv-2006\ca.jpg', dpi=300)
plt.show()

# Process CSV files
all_data = []
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        parts = file_name.split('_')
        latitude = float(parts[1])
        longitude = float(parts[2])
        type = str(parts[4])
        if 33.4 <= latitude <= 34.5 and -119.4 <= longitude <= -116.8 and type == 'DPV':
            df = pd.read_csv(os.path.join(folder_path, file_name))
            all_data.append(df)

all_data_15min = []
for data in all_data:
    x = data['Power(MW)']
    x = np.array(x)
    if len(x) != 288 * 365:
        print(f"Skipping file due to unexpected length: {len(x)}")
        continue
    x = x.reshape(365, 288)  # Each day has 288 5-minute intervals
    x_15min = x.reshape(365, 96, 3).sum(axis=2)  # Sum every three 5-minute intervals to get 15-minute intervals
    all_data_15min.append(x_15min)

all_data_15min = np.array(all_data_15min)
print(all_data_15min.shape)
np.save(folder_path + r'\ca_data_99solar_15min.npy', all_data_15min)
