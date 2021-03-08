
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import torch
import torch.autograd as autograd # torch中自动计算梯度模块
import torch.nn as nn             # 神经网络模块
import torch.nn.functional as F   # 神经网络模块中的常用功能 
import torch.optim as optim       # 模型优化器模块

emb1=[]
emb2=[]
emb1=[1,2,3]
emb2=[1,2,3]

 
res=np.multiply(emb1, emb2)

print(res)