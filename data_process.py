# -*- coding: utf-8 -*-
import os
from scipy.io import loadmat,savemat
import numpy as np
from sklearn.model_selection import train_test_split

# In[]
lis=os.listdir('./dataset/cwru/0HP/')

N=100;Len=1024

data=np.zeros((0,Len))
label=[]
for n,i in enumerate(lis):
    path='./dataset/cwru/0HP/'+i
    print('第',n,'类的数据是',path,'这个文件')
    file=loadmat(path)
    file_keys = file.keys()
    for key in file_keys:
        if 'DE' in key:
            files= file[key].ravel()
    data_=[]
    for i in range(N):
        start=np.random.randint(0,len(files)-Len)
        end=start+Len
        data_.append(files[start:end])
        label.append(n)
    data_=np.array(data_)

    data=np.vstack([data,data_])
label=np.array(label)

# 划分出训练数据集
train_x, temp_x, train_y, temp_y = train_test_split(data, label, test_size=0.4, random_state=0)

# 将剩余的数据等分为验证数据集和测试数据集
val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=0)

# In[] 保存数据
savemat('result/data_process_image.mat',{'train_x':train_x,'test_x':test_x,'val_x':val_x,
                                   'train_y':train_y,'test_y':test_y,'val_y':val_y})
