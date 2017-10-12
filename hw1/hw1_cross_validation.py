import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import pandas as pd

data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
    data.append([])

n_row = 0
train_path ='C:/Users/TappyHsieh/Desktop/train.csv'
# 'C:/Users/TappyHsieh/Desktop/train.csv'
text = open(train_path, 'r', encoding='big5')
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3, 27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))
    n_row = n_row+1
text.close()

x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s])
        y.append(data[9][480*i+j+9])

x = np.array(x)
y = np.array(y)
X = [1] * len(x)
for row in range(len(x)):
    X[row] = []
# pick PM10 PM2.5 RH
for row in range(len(x)):
    X[row] = x[row][np.r_[
                         #3,4,5,6,7,8,
                         #12,13,14,15,16,17,
						 #21,22,23,24,25,26,
						 #30,31,32,33,34,35,
						 #39,40,41,42,43,44,
						 #48,49,50,51,52,53,
						 #57,58,59,60,61,62,
						 #66,67,68,69,70,71,
						 #72,73,74,75,76,77,78,79,80,
                         81,82,83,84,85,86,87,88,89,
						 #98,
						 #99,100,101,102,103,104,105,106,107,
						 #111,112,113,114,115,116,
						 #120,121,122,123,124,125,
						 #129,130,131,132,133,134,
						 #138,139,140,141,142,143,
						 #137,148,149,150,151,152,
                         #156,157,158,159,160,161
                         ]]

x = np.array(X)


x_transpose = list(map(list, zip(*x)))
Varlist = []
Minlist = []
for row in range(x.shape[1]):
    Varlist.append(np.std(x_transpose[row]))
    Minlist.append(np. mean(x_transpose[row]))

for row in range(x.shape[0]):
    for col in range(x.shape[1]):
        x[row][col] = (x[row][col] - Minlist[col]) / Varlist[col]

# add square term
x = np.concatenate((x,x**2), axis=1)
# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)


l_rate = 10
repeat = 10000
Lambda = 0.001

## Linear regression


def Cal_MSE(x_training_data, x_testing_data, y_training_data, y_testing_data):
    w = np.zeros(len(x_training_data[0]))

    x_t = x_training_data.transpose()
    s_gra = np.zeros(len(x_training_data[0]))

    for i in range(repeat):
        hypo = np.dot(x_training_data, w)
        loss = hypo - y_training_data #+ Lambda * np.sum(w**2)
        cost = np.sum(loss**2) / len(x_training_data) 
        cost_a = math.sqrt(cost)
        gra = np.dot(x_t, loss)
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        w = w - l_rate * gra / ada
        if i % 100 ==0:
            print ('iteration: %d | Cost: %f  ' % (i, cost_a))

    '''
    # save model
    np.save('model_valid.npy',w)
    # read model
    w = np.load('model_valid.npy')
    '''

    ans = []
    y_hat = []
    for i in range(len(x_testing_data)):
        ans.append(["id_"+str(i)])
        a = np.dot(w, x_testing_data[i])
        ans[i].append(a)
        y_hat.append(a)
    error = y_testing_data-np.array(y_hat)
    RMSE = np.sqrt(np.sum(error**2)/len(y_testing_data))
    return RMSE


MSE1 = Cal_MSE(x[1501:], x[0:1500], y[1501:], y[0:1500])
MSE2 = Cal_MSE(x[np.r_[0:1500, 3001::]], x[1501:3000],
               y[np.r_[0:1500, 3001::]], y[1501:3000])
MSE3 = Cal_MSE(x[np.r_[0:3001, 4501::]], x[3001:4500],
               y[np.r_[0:3001, 4501::]], y[3001:4500])
MSE4 = Cal_MSE(x[0:4499], x[4500::], y[0:4499],  y[4500::])

print(MSE1, MSE2, MSE3, MSE4)
print((MSE1 + MSE2 + MSE3+MSE4)/4)

