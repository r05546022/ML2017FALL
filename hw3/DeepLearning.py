import csv
import math
import random
import sys

import numpy as np
import pandas as pd
from numpy.linalg import inv
#from PIL import Image
# Variable setting
import keras
from keras.models import *
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, ThresholdedReLU, LeakyReLU, AveragePooling2D, ZeroPadding2D

#### Set file path
'''
train_path = 'C:/Users/TappyHsieh/Desktop/train.csv'
test_path = 'C:/Users/TappyHsieh/Desktop/test.csv'
filename = "C:/Users/TappyHsieh/Desktop/predictdata.csv"

train_path = 'C:/Users/argon/Desktop/train.csv'
test_path = 'C:/Users/argon/Desktop/test.csv'
filename = "C:/Users/argon/Desktop/predictdata.csv"
feature_path = "C:/Users/argon/Desktop/feature/feature.npy"
label_path = "C:/Users/argon/Desktop/feature/label.npy"
feature_test_path = "C:/Users/argon/Desktop/feature/feature_test.npy"

'''
train_path = 'C:/Users/user/Desktop/train.csv'
test_path = 'C:/Users/user/Desktop/test.csv'
filename = "C:/Users/user/Desktop/predictdata.csv"
feature_path = "C:/Users/user/Desktop/feature/feature.npy"
label_path = "C:/Users/user/Desktop/feature/label.npy"
feature_test_path = "C:/Users/user/Desktop/feature/feature_test.npy"



test_data = []   # 7178
test_y = []      # 7178
train_data = []  # 28791
train_y = []     # 28791

'''
#  Read Data
#### Read training data
df = pd.read_csv(train_path)
y = df['label'].as_matrix()
y = pd.get_dummies(y).values
X = df['feature'].as_matrix()
X = np.array([np.array([*map(int, x.split())]).reshape(48, 48) for x in X])
np.save(feature_path, X)
np.save(label_path, y)

#### Read testing data
df = pd.read_csv(test_path)
y_test = df['id'].as_matrix()
y_test = pd.get_dummies(y_test).values
X_test = df['feature'].as_matrix()
X_test = np.array([np.array([*map(int, x.split())]).reshape(48, 48) for x in X_test])
np.save(feature_test_path, X_test)
'''
X = np.load(feature_path)
y = np.load(label_path)
X_test = np.load(feature_test_path)

def build_model():
    input_img = Input(shape=(48, 48, 1))
    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)
    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)
    block4 = Conv2D(128, (3, 3), activation='relu')(block3)

    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)
    block5 = Conv2D(128, (3, 3), activation='relu')(block4)

    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)

    block5 = Flatten()(block5)
    fc1 = Dense(512, activation='relu')(block5)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(512, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)
    fc3 = Dense(512, activation='relu')(fc1)
    fc3 = Dropout(0.5)(fc2)
    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)
    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=1e-3)
    # opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

X = X.astype('float32')
X /= 255
X = X.reshape(len(X), 48, 48, 1)
'''
model = Sequential()
#model.add(Conv2D(49, kernel_size=(7, 7), activation=ThresholdedReLU(theta=1.0), input_shape=(48, 48, 1), padding='same'))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu',input_shape=(48, 48, 1), padding='valid'))
model.add(ZeroPadding2D(padding=(2, 2)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D(padding=(1, 1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
            #,padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(ZeroPadding2D(padding=(1, 1)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=1./20))
#model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(1024, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
'''
model = build_model()
model.fit(X, y,
          batch_size=300, epochs=50,
          verbose=1, validation_split=0.1, class_weight='auto')


X_test = X_test.astype('float32')
X_test /= 255
X_test = X_test.reshape(len(X_test), 48, 48, 1)
preds = model.predict(X_test)

result = []
for row in range(len(preds)):
    result.append(np.argmax(preds[row]))

text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(result)):
    s.writerow([i,result[i]])

text.close()


'''
def visualize(X):
    	im = Image.fromarray(X)
    im.show()
'''
