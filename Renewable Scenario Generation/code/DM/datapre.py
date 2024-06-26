# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:30:09 2023

@author: yuxin dai
@e-mail:yuxindai@whu.edu.cn
"""

import numpy as np
path=r'C:\Users\Desktop1\Desktop\边界样本生成\data'
data_S=np.load(path+r'\S_sample200000713.npy')
data_US=np.load(path+r'\US_sample200000713.npy')

data_all=np.append(data_S,data_US,axis=2)
#data_all=data_all[:15000,:]
np.save(path+r'\dataall.npy',data_all)
#%%
i=0
while i<100:
    dist=np.linalg.norm(data_US[i,:,-1] - data_S[i,:,-1])
    print(dist)
    i=i+1