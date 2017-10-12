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
                     #  18, 19, 20, 21, 22, 23, 24, 25, 26,
                     #  27, 28, 29, 30, 31, 32, 33, 34, 35,
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

test_path = sys.argv[1] #'C:/Users/TappyHsieh/Desktop/test.csv'
filename = sys.argv[2] #"C:/Users/TappyHsieh/Desktop/predictdata.csv" 
# read model
w = np.load('model_best.npy')

test_x = []
n_row = 0
text = open(test_path, "r")
row = csv.reader(text , delimiter= ",")
for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

test_X = [1] * len(test_x)
# pick Variable
for row in range(len(test_x)):
    test_X[row] = test_x[row][Variable_list]
test_x = np.array(test_X)
# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)
# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()
