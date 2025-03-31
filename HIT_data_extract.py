import numpy as np
import scipy.io as sio

# data1 = np.load('dataset/data1.npy')
# data2 = np.load('dataset/data2.npy')
# data3 = np.load('dataset/data3.npy')
# data4 = np.load('dataset/data4.npy')
# data5 = np.load('dataset/data5.npy')

data1 = np.load('dataset/hit/data1.npy')
data3 = np.load('dataset/hit/data3.npy')
data5 = np.load('dataset/hit/data5.npy')

row5_data1 = data1[:, 4, :]  # 形状为(504, 20480)
row5_data3 = data3[:, 4, :]  # 形状为(504, 20480)
row5_data5 = data5[:, 4, :]  # 形状为(450, 20480)

col_data1 = np.ravel(row5_data1)  # 形状为(10321920,)
col_data3 = np.ravel(row5_data3)  # 形状为(10321920,)
col_data5 = np.ravel(row5_data5)  # 形状为(9216000,)

col_data_label_0 = col_data1
col_data_label_1 = col_data1
col_data_label_2 = col_data5

sio.savemat('data_HIT/data0.mat', {'label_0': col_data_label_0})
sio.savemat('data_HIT/data1.mat', {'label_1': col_data_label_1})
sio.savemat('data_HIT/data2.mat', {'label_2': col_data_label_2})












