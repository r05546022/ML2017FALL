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

train_path = 'C:/Users/user/Desktop/train.csv'
test_path = 'C:/Users/user/Desktop/test.csv'
filename = "C:/Users/user/Desktop/predictdata.csv"
feature_path = "C:/Users/user/Desktop/feature/feature.npy"
label_path = "C:/Users/user/Desktop/feature/label.npy"
label_array_path = "C:/Users/user/Desktop/feature/label_array.npy"
feature_test_path = "C:/Users/user/Desktop/feature/feature_test.npy"


# test_X = []      # 7178
# test_y = []      # 7178
# train_X = []     # 28791
# train_y = []     # 28791


X = np.load(feature_path)
y = np.load(label_path)

X = X.astype('float32')
X /= 255
X = X.reshape(len(X), 48, 48, 1)

from sklearn.utils import shuffle
col_list = np.array(list(range(len(X))))
X,y, col_list = shuffle(X,y, col_list, random_state=0)


X, y, col_list = shuffle(X, y, col_list, random_state=0)

train_X = X[0:int(len(X) * 0.9)]
train_Y = y[0:int(len(y) * 0.9)]
test_X = X[(int(len(X) * 0.9)+1):len(X)]
test_Y = y[(int(len(y) * 0.9)+1):len(y)]


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
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(AveragePooling2D((2,2), padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512, kernel_regularizer=regularizers.l2(0.02)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(train_X, train_Y, batch_size=500,
          epochs=100, verbose=1, class_weight='auto')
model.save('Model_seperate_Training_Testing_3559431_par.h5')

model = load_model('Model_seperate_Training_Testing_3559431_par.h5')
self_preds = model.predict(test_X)
self_preds = self_preds.argmax(axis=-1)
label_array = test_Y.argmax(axis=-1)


from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import *
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(
            cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


conf_mat = confusion_matrix(label_array,self_preds)

plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
plt.show()

