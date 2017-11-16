import csv
import math
import random
import sys
import os
import numpy as np
import pandas as pd
from numpy.linalg import inv
# Variable setting
import keras
from keras.models import *
from keras.models import load_model
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, ThresholdedReLU, LeakyReLU, AveragePooling2D, ZeroPadding2D

#### Set file path
test_path = sys.argv[1]
print(" test path : ", test_path)
filename = sys.argv[2]
print(" filename path : ", filename)
#wget https://github.com/r05546022/ML2017FALL/releases/download/0.0.1/Model.h5

# Read testing data
df = pd.read_csv(test_path)
X_test = df['feature'].as_matrix()
X_test = np.array([np.array([*map(int, x.split())]).reshape(48, 48) for x in X_test])
X_test = X_test.astype('float32')
X_test /= 255
X_test = X_test.reshape(len(X_test), 48, 48, 1)
try:
    model = load_model('Model.h5')
    print("model downloaded and imported")
except:
    os.system('wget https://github.com/r05546022/ML2017FALL/releases/download/0.0.1/Model.h5')
    model = load_model('Model.h5')

preds = model.predict(X_test)

result = []
for row in range(len(preds)):
    result.append(np.argmax(preds[row]))

text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(result)):
    s.writerow([i, result[i]])
text.close()

'''
# python hw3_test.py C:/Users/user/Desktop/test.csv C:/Users/user/Desktop/Predictdata.csv
test_path = 'C:/Users/user/Desktop/test.csv'
print(" test path : ", test_path)
filename = 'C:/Users/user/Desktop/Predictdata.csv'
print(" filename path : ", filename)

bash hw3_test.sh C:/Users/user/Desktop/test.csv C:/Users/user/Desktop/Predictdata.csv
'''
