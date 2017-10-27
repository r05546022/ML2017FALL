import csv
import math
import random
import sys
import xgboost as xgb

import numpy as np
import pandas as pd
from numpy.linalg import inv

#### Set file path
X_train_path = sys.argv[1]
Y_train_path = sys.argv[2]
X_test_path = sys.argv[3]
fileName = sys.argv[4]

test_data = []  #16281   106
data = []  #32561   106
Y = []

#  Read Data

#### Read X_test data
for i in range(16282):
    test_data.append([])
n_row = 0
text = open(X_test_path, 'r')
row = csv.reader(text, delimiter = ",")
for r in row:
    test_data[n_row] = r
    n_row += 1
text.close()

#### Read X_train data
data = []  #32561   106
for i in range(32562):
    data.append([])
n_row = 0
text = open(X_train_path, 'r')#, encoding = 'big5')
row = csv.reader(text, delimiter = ",")
for r in row:
    data[n_row] = r
    n_row += 1
text.close()

#### Read Y_train data
Y = []
text = open(Y_train_path, 'r')#, encoding = 'big5')
row = csv.reader(text, delimiter = ",")
Y = [r for r in csv.reader(text, delimiter = ",")]

# set the data into np.array
y = []
for row in range(1,len(data)):
    y.append(float(Y[row][0]))
    for col in  range(len(data[0])):
        data[row][col] = float(data[row][col])
x = np.array(data[1::])
y = np.array(y)

# set the test data into np.array
for row in range(1,len(test_data)):
    for col in  range(len(test_data[0])):
        test_data[row][col] = float(test_data[row][col])
x_test = np.array(test_data[1::])

from xgboost.sklearn import XGBClassifier

# Parameter setting
param = {}
param['objective'] = 'binary:logistic'
param['gamma'] = 0
param['max_depth'] = 6
param['min_child_weight']=1
param['max_delta_step'] = 0
param['subsample']= 1
param['colsample_bytree']=1
param['silent'] = 1
param['seed'] = 0
param['base_score'] = 0.5


xclas = XGBClassifier(**param)
xclas.fit(x, y)
result = xclas.predict(x_test)

text = open(fileName, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(result)):
    s.writerow([i+1,int(result[i])])
text.close()

#    X_train_path = '/home/tappy/X_train'
#    Y_train_path ='/home/tappy/Y_train'
#    X_test_path ='/home/tappy/X_test'
#    fileName = '/home/tappy/predictdata.csv'
#    # X_train_path = 'C:/Users/TappyHsieh/Desktop/X_train'
#    # Y_train_path = 'C:/Users/TappyHsieh/Desktop/Y_train'
#    # X_test_path = 'C:/Users/TappyHsieh/Desktop/X_test
#    # filename = "C:/Users/user/Desktop/predictdata.csv"
#
#
#    # Standatdize data
#    Xt = np.transpose(x)
#    X_nuimerical_list = np.r_[0,1,3,5]
#    X_nuimerical_variable=Xt[X_nuimerical_list]
#    mean_nuimerical_variable = [np.mean(row) for row in X_nuimerical_variable ]
#    std_nuimerical_variable = [np.std(row) for row in X_nuimerical_variable ]
#    for row in range(len(X_nuimerical_list)) :
#        for col in range(len(Xt[X_nuimerical_list[row]])):
#            Xt[X_nuimerical_list[row]][col] = ( Xt[X_nuimerical_list[row]][col] - mean_nuimerical_variable[row] )/std_nuimerical_variable[row]
#    X = np.transpose(Xt)
