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

#### Set file path
train_path = sys.argv[1]

#  Read Data
#### Read training data
df = pd.read_csv(train_path)
y = df['label'].as_matrix()
label_array = df['label'].as_matrix()
y = pd.get_dummies(y).values
X = df['feature'].as_matrix()
X = np.array([np.array([*map(int, x.split())]).reshape(48, 48) for x in X])
X = X.astype('float32')
X /= 255
X = X.reshape(len(X), 48, 48, 1)

#  Set model
model = Sequential()

model.add(Conv2D(64, (5, 5), input_shape=(48, 48, 1)))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3)))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.35))

model.add(Conv2D(512, (3, 3)))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3)))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(AveragePooling2D((2,2), padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512, kernel_regularizer=regularizers.l2(0.02),activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(512,activation = 'relu'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(X, y,batch_size=700, epochs=180,verbose=1, class_weight='auto')
model.save('Model.h5')