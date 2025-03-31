# -*- coding: utf-8 -*-
# 对所有样本依次计算时频图 并保存
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.io import loadmat
import os
import shutil
shutil.rmtree('image/')
os.mkdir('image/')
os.mkdir('image/train')
os.mkdir('image/test')
os.mkdir('image/val')
# In[]
def Spectrum(data,label,path):
    label=label.reshape(-1,)
    for i in range(data.shape[0]):
        sampling_rate=48000# 采样频率
        wavename = 'cmor3-3'#cmor是复Morlet小波，其中3－3表示Fb－Fc，Fb是带宽参数，Fc是小波中心频率。
        totalscal = 256
        fc = pywt.central_frequency(wavename)   # 小波的中心频率
        cparam = 2 * fc * totalscal
        scales = cparam / np.arange(totalscal, 1, -1)
        [cwtmatr, frequencies] = pywt.cwt(data[i], scales, wavename, 1.0 / sampling_rate)
        t=np.arange(len(data[i]))/sampling_rate

        t,frequencies = np.meshgrid(t,frequencies)
        plt.pcolormesh(t, frequencies, abs(cwtmatr),cmap='jet')    
        plt.axis('off')
        plt.savefig(path+'/'+str(i)+'_'+str(label[i])+'.jpg', bbox_inches='tight',pad_inches = 0)
        plt.close()
data=loadmat('result/data_process_image.mat')
Spectrum(data['train_x'],data['train_y'],'image/train')
Spectrum(data['test_x'],data['test_y'],'image/test')
Spectrum(data['val_x'],data['val_y'],'image/val')
