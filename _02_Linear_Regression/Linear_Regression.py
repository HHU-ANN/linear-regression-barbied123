# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data,a=0.1):
    x,y=read_data()
    weight=np.matmul(np.linalg.inv(np.matmul(x.T,x)+np.eye(x.shape[1]*a)),np.matmul(x.T,y))
    return weight@data
    
def lasso(data,t=1,a=0.1):
    x,y=read_data()
    for i in range(400):
        M=w-wt*np.matmul(x.T,x)+t*np.matmul(x.T,y)
        w=sgn(M)(abs(M)-at)
    return weight@data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
