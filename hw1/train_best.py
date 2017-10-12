import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import pandas as pd

# Variable_list
Variable_list = np.r_[
                     #   0,  1,  2,  3,  4,  5,  6,  7,  8,
                     #   9, 10, 11, 12, 13, 14, 15, 16, 17,
                     #  27, 28, 29, 30, 31, 32, 33, 34, 35,
                     #  18, 19, 20, 21, 22, 23, 24, 25, 26,
                     #  36, 37, 38, 39, 40, 41, 42, 43, 44,
                     #  45, 46, 47, 48, 49, 50, 51, 52, 53,
                     #  54, 55, 56, 57, 58, 59, 60, 61, 62,
                     #  63, 64, 65, 66, 67, 68, 69, 70, 71,
                     #  72, 73, 74, 75, 76, 77, 78, 79, 80,
                        81, 82, 83, 84, 85, 86, 87, 88, 89,
                     #  90, 91, 92, 93, 94, 95, 96, 97, 98,
                     #  99,100,101,102,103,104,105,106,107,
                     # 108,109,110,111,112,113,114,115,116,
                     # 117,118,119,120,121,122,123,124,125,
                    # 126,127,128,129,130,131,132,133,134,
                     # 135,136,137,138,139,140,141,142,143,
                     # 144,145,146,147,148,149,150,151,152,
                     # 153,154,155,156,157,158,159,160,161
                     ]

train_path = sys.argv[1] #'C:/Users/TappyHsieh/Desktop/train.csv' #'/home/tappy/train.csv' 
# test_path = 'C:/Users/TappyHsieh/Desktop/test.csv'
# filename = sys.argv[2] #"C:/Users/TappyHsieh/Desktop/predictdata.csv" # '/home/tappy/predictdata.csv'  #

def main():
    l_rate = 10
    repeat = 12000
    Lambda = 0.005
    
    
    data = []
    for i in range(18):
        data.append([])
    n_row = 0
    text = open(train_path, 'r' )#, encoding='big5')
    row = csv.reader(text , delimiter=",")
    for r in row:
        if n_row != 0:
            for i in range(3, 27):
                if r[i] != "NR":
                    data[(n_row-1)%18].append(float(r[i]))
                else:
                    data[(n_row-1)%18].append(float(0))
        n_row = n_row + 1
    text.close()
    x = []
    y = []
    for i in range(12):
        for j in range(471):
            x.append([])
            for t in range(18):
                for s in range(9):
                    x[471*i+j].append(data[t][480*i+j+s] )
            y.append(data[9][480*i+j+9])
    x = np.array(x)
    y = np.array(y)
    # pick Variable
    X = [1] * len(x)
    for row in range(len(x)):
        X[row] = x[row][Variable_list]
    x = np.array(X)

    '''
    x_transpose = list(map(list, zip(*x)))
    Varlist = []
    Minlist = []
    for row in range(x.shape[1]):
        Varlist.append(np.std(x_transpose[row]))
        Minlist.append(np.mean(x_transpose[row]))
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            x[row][col] = (x[row][col] - Minlist[col]) / Varlist[col]
    '''
    # add square term
    x = np.concatenate((x,x**2), axis=1)
    # add bias
    x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
    w = np.zeros(len(x[0]))
    ## Linear regression
    x_t = x.transpose()
    s_gra = np.zeros(len(x[0]))
    for i in range(repeat):
        hypo = np.dot(x, w)
        loss = hypo - y  + Lambda * np.sum(w**2)
        cost = np.sum(loss**2) / len(x)
        cost_a = math.sqrt(cost)
        gra = np.dot(x_t, loss)
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        w = w - l_rate * gra / ada
        if i%100==0:
        	print ('iteration: %d | Cost: %f ' % (i, cost_a))
    # save model
    np.save('model_best.npy',w)

if __name__ == '__main__':
    main()
