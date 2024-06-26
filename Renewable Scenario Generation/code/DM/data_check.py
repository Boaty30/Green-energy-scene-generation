# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:41:43 2023

@author: yuxin dai
@e-mail:yuxindai@whu.edu.cn
"""

import numpy as np
from pypower.api import case9, ppoption, runpf
path=r'C:\Users\Desktop1\Desktop\边界样本生成\data'
dataall=np.load(path+r'\dataall_delete.npy')
#%%
def data_input(data,ppc):
    #修改节点注入有功功率
    ppc['gen'][1,1]=data[0][0]
    ppc['gen'][2,1]=data[1][0]
    ppc['bus'][4,2]=data[2][0]
    ppc['bus'][6,2]=data[3][0]
    ppc['bus'][8,2]=data[4][0]    
    return ppc
def state_check(pf_results):
    volt_bus=pf_results[0]['bus'][:,7]
    volt_bus_max=pf_results[0]['bus'][:,11]
    volt_bus_min=pf_results[0]['bus'][:,12]
    
    p_gen=pf_results[0]['gen'][:,1]
    p_gen_max=pf_results[0]['gen'][:,8]
    p_gen_min=pf_results[0]['gen'][:,9]
    
    q_gen=pf_results[0]['gen'][:,2]
    q_gen_max=pf_results[0]['gen'][:,3]
    q_gen_min=pf_results[0]['gen'][:,4]
    
    p_branch=pf_results[0]['branch'][:,13]
    p_branch_max=pf_results[0]['branch'][:,7]
    
    p_load=pf_results[0]['bus'][[[4,6,8]],2].reshape(3,1)
    
    
    label_bus=(volt_bus>=volt_bus_min).all() and (volt_bus<=volt_bus_max).all()
    #print("label_bus",label_bus,volt_bus,volt_bus>=volt_bus_min,volt_bus<=volt_bus_max)
    label_p_gen=(p_gen>=p_gen_min).all() and (p_gen<=p_gen_max).all()
    label_q_gen=(q_gen>=q_gen_min).all() and (q_gen<=q_gen_max).all()
    
    label_branch=(np.abs(p_branch)<=p_branch_max).all()
    
    label_load=(p_load>0).all()
    
    label=[label_bus,label_p_gen,label_q_gen,label_branch,label_load]
    
    
    #label=0 安全，1母线电压越限，2发电机有功越限，3，发电机无功越限，4支路热稳越限,5负荷值异常,6潮流不收敛
    label_pfconverge=pf_results[1] == 1
    if label_pfconverge:
        if False in label:
            return label.index(False)+1
        else: 
            return 0
    else:
        return 6
def data_check(data):
    #data的shape为(5,2)
    data_S=data[:,0].reshape(5,1)
    data_U=data[:,1].reshape(5,1)
    ppc=case9()
    ppopt=ppoption(PF_ALG=1)
    
    ppc=data_input(data_S,ppc)
    pf_results=runpf(ppc,ppopt)
    label_S=state_check(pf_results)
    
    ppc=data_input(data_U,ppc)
    pf_results=runpf(ppc,ppopt)
    label_U=state_check(pf_results)
    
    if label_S==0 and label_U !=0 :
        if np.linalg.norm( data_S- data_U) <=10.1:
            return 0 #样本有效
        else:
            return 1 #样本无效,距离超过阈值
    else:
        return 2 #样本无效,样本对不符合样本对定义

for i in range(15000):
    state=data_check(dataall[i,])
    print(state)


    
    

