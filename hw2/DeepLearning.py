import csv
import math
import random
import sys
import xgboost as xgb

import numpy as np
import pandas as pd
from numpy.linalg import inv

# Variable setting


#### Set file path

X_train_path = 'C:/Users/TappyHsieh/Desktop/X_train'#'/home/tappy/train.csv'  #
Y_train_path = 'C:/Users/TappyHsieh/Desktop/Y_train'
X_test_path = 'C:/Users/TappyHsieh/Desktop/X_test'#'/home/tappy/train.csv'  #
filename = "C:/Users/TappyHsieh/Desktop/predictdata.csv"# '/home/tappy/predictdata.csv'  #

'''
X_train_path = "C:/Users/user/Dropbox/Course/gadurate courses/Machine Learning/hw2/feature/X_train"
Y_train_path = "C:/Users/user/Dropbox/Course/gadurate courses/Machine Learning/hw2/feature/Y_train"
X_test_path = "C:/Users/user/Dropbox/Course/gadurate courses/Machine Learning/hw2/feature/X_test"
filename = "C:/Users/user/Desktop/predictdata.csv"
'''
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

# Set the training data into np.array
y = []
for row in range(1,len(data)):
    y.append(int(Y[row][0]))
    for col in  range(len(data[0])):
        data[row][col] = float(data[row][col])
x = np.array(data[1::])
y = np.array(y)
# Set X_test data into np.array
for row in range(1,len(test_data)):
    for col in  range(len(test_data[0])):
        test_data[row][col] = float(test_data[row][col])
x_test = np.array(test_data[1::])



# Standatdize data
Xt = np.transpose(x)
X_nuimerical_list = np.r_[0,1,3,5]
X_nuimerical_variable=Xt[X_nuimerical_list]
mean_nuimerical_variable = [np.mean(row) for row in X_nuimerical_variable ]
std_nuimerical_variable = [np.std(row) for row in X_nuimerical_variable ]
for row in range(len(X_nuimerical_list)) :
    for col in range(len(Xt[X_nuimerical_list[row]])):
        Xt[X_nuimerical_list[row]][col] = ( Xt[X_nuimerical_list[row]][col] - mean_nuimerical_variable[row] )/std_nuimerical_variable[row]
X = np.transpose(Xt)

'''
# Standatdize X_train data
Xt = np.transpose(x)
mean_X_train = [np.mean(row) for row in Xt ]
std_X_train = [np.std(row) for row in Xt ]
for row in range(len(Xt)) :
    for col in range(len(Xt[0])):
        Xt[row][col] = (Xt[row][col]-mean_X_train[row])/std_X_train[row]
X = np.transpose(Xt)
# Standardize X_test data
X_test_t = np.transpose(x_test)
mean_X_test = [np.mean(row) for row in X_test_t ]
std_X_test = [np.std(row) for row in X_test_t ]
for row in range(len(X_test_t)) :
    for col in range(len(X_test_t[0])):
        Xt[row][col] = ( X_test_t[row][col] - mean_X_test[row] )/std_X_test[row]
X_test = np.transpose(X_test_t)
'''
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras import regularizers

model = Sequential()

x_train = X
x_test = x_test
y_train = y
#np_utils.to_categorical(y)

#
#model.add(Dense(units=1000, input_dim=106, activation = 'relu'))
#model.add(Dropout(0.5))
#model.add(Dense(units=1, activation = 'relu'))
#model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
#model.fit(x_train, y_train, epochs=5, batch_size=32,class_weight='auto')

model = Sequential()
model.add(Dense(input_dim=x_train.shape[1],units =1024,activation='relu',kernel_regularizer=regularizers.l1(0.001)))
model.add(Dropout(0.1))
model.add(Dense(units =1024,activation='sigmoid'))
model.add(Dense(units=1,activation='hard_sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,verbose=1,batch_size=500,epochs=15,validation_split=0.3,class_weight='auto')


preds = model.predict(x_test)

result = []
for row in range(len(preds)):
	if preds[row][0] < 0.5:
		result.append(0)
	else:
		result.append(1)

text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(result)):
    s.writerow([i+1,result[i]])
text.close()

'''
#bst.dump_model('C:/Users/TappyHsieh/Desktop/dump.raw.txt')
preds = []
result = []
for row in range(len(preds)):
	if preds[row][0] > preds[row][1]:
		result.append(0)
	else:
		result.append(1)


text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(result)):
    s.writerow([i+1,result[i]])
text.close()


'''



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
