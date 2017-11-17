#!/usr/bin/env python
# -- coding: utf-8 --

import csv
import math
import random
import sys
import numpy as np
import pandas as pd
from numpy.linalg import inv
import keras
from keras.models import *
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, ThresholdedReLU, LeakyReLU, AveragePooling2D, ZeroPadding2D

import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import *
import itertools

#### Set file path
train_path = 'C:/Users/user/Desktop/train.csv'
test_path = 'C:/Users/user/Desktop/test.csv'


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
#  Read Data
#### Read training data
df = pd.read_csv(train_path)
y = df['label'].as_matrix()
y = pd.get_dummies(y).values
X = df['feature'].as_matrix()
X = np.array([np.array([*map(int, x.split())]) for x in X])
X = X.astype('float32')
X /= 255

model = Sequential()
model.add(Dense(input_dim=X.shape[1], units=1024,
                activation='relu', kernel_regularizer=regularizers.l1(0.001)))
model.add(Dropout(0.1))
model.add(Dense(512, kernel_regularizer=regularizers.l2(0.02)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(128, kernel_regularizer=regularizers.l2(0.02)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(256, kernel_regularizer=regularizers.l2(0.02)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.summary()
train_history = model.fit(X, y, batch_size=8000, epochs=400,
                          verbose=1, validation_split=0.1, class_weight='auto')

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')
'''
_________________________________________________________________
Layer(type)                 Output Shape              Param
== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
dense_118(Dense)(None, 1024)              2360320
_________________________________________________________________
dropout_78(Dropout)(None, 1024)              0
_________________________________________________________________
dense_119(Dense)(None, 512)               524800
_________________________________________________________________
activation_78(Activation)(None, 512)               0
_________________________________________________________________
dropout_79(Dropout)(None, 512)               0
_________________________________________________________________
dense_120(Dense)(None, 512)               262656
_________________________________________________________________
activation_79(Activation)(None, 512)               0
_________________________________________________________________
dropout_80(Dropout)(None, 512)               0
_________________________________________________________________
dense_121(Dense)(None, 128)               65664
_________________________________________________________________
activation_80(Activation)(None, 128)               0
_________________________________________________________________
dropout_81(Dropout)(None, 128)               0
_________________________________________________________________
dense_122(Dense)(None, 256)               33024
_________________________________________________________________
activation_81(Activation)(None, 256)               0
_________________________________________________________________
dropout_82(Dropout)(None, 256)               0
_________________________________________________________________
dense_123(Dense)(None, 256)               65792
_________________________________________________________________
activation_82(Activation)(None, 256)               0
_________________________________________________________________
dropout_83(Dropout)(None, 256)               0
_________________________________________________________________
dense_124(Dense)(None, 128)               32896
_________________________________________________________________
activation_83(Activation)(None, 128)               0
_________________________________________________________________
dropout_84(Dropout)(None, 128)               0
_________________________________________________________________
dense_125(Dense)(None, 7)                 903
_________________________________________________________________
activation_84(Activation)(None, 7)                 0
=================================================================
Total params: 3, 346, 055
Trainable params: 3, 346, 055

Epoch 180/180
25838/25838 [==============================] - 0s 16us/step
- loss: 2.1008 - acc: 0.2516 - val_loss: 2.0967 - val_acc: 0.2483

'''
