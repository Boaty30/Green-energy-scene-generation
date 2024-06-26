# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 16:47:18 2023

@author: yuxin dai
@e-mail:yuxindai@whu.edu.cn
"""

import torch
x = torch.randn(32,1,5,2)


conv = torch.nn.Conv2d(1,4,(3,3),padding=1)
conv1 = torch.nn.Conv2d(4,4,(1,4),padding=1)
res = conv(x)
print(res.shape) 
res=conv1(res)
print(res.shape) 
#%%
import numpy as np
from sklearn.preprocessing import MinMaxScaler
path=r'C:\Users\Desktop1\Desktop\边界样本生成\data'
data1=np.load(path+r'\dataall_delete.npy')

data=np.reshape(data1,(15000,10,1))
#%%
# min_max_scaler = MinMaxScaler()
# min_max_x = min_max_scaler.fit_transform(data)
# min_max_x=np.reshape(min_max_x,(15000,5,2))

data_max=np.max(data,0)
data_min=np.min(data,0)
data_minmax=(data_max-data)/(data_max-data_min)
data_save=np.reshape(data_minmax,(15000,2,5))
data_save1=np.transpose(data_save, axes=[0, 2, 1])
np.save(path+r'\data_minmax.npy',data_save1,allow_pickle=True)
#%%


