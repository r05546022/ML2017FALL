#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, TimeDistributed, Bidirectional
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import *
from sklearn.model_selection import train_test_split

MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

batchsize = 500
num_epoch = 4

####################
training_label = 'c:/Users/user/Desktop/training_label.txt'
training_nolabel = 'c:/Users/user/Desktop/training_nolabel.txt'
testing = 'c:/Users/user/Desktop/testing_data.txt'

train_text_path = 'c:/Users/user/Desktop/feature/train_text.npy'
y_train_path = 'c:/Users/user/Desktop/feature/y_train.npy'
valid_text_path = 'c:/Users/user/Desktop/feature/valid_text.npy'
test_text_path = 'c:/Users/user/Desktop/feature/test_text.npy'

predict_path = 'c:/Users/user/Desktop/result/predict.csv'
filepath = "c:/Users/user/Desktop/weight/weights-improvement-{epoch:02d}-{val_acc:.3f}.hdf5"
good_incidence_path = 'c:/Users/user/Desktop/feature/good_incidence.npy'
####################


def load(readnpy=False):
    if readnpy:
        y_train = np.load('./feature/y_train.npy')
        train_text = np.load('./feature/train_text.npy')
        test_text = np.load('./feature/test_text.npy')
    else:
        with open('./training_label.txt', "U", encoding='utf-8-sig') as f:
            y_train = []
            train_text = []
            test_text = []
            for l in f:
                y_train.append(l.strip().split("+++$+++")[0])
                train_text.append(l.strip().split("+++$+++")[1])
            np.save('./feature/y_train.npy', y_train)
            np.save('./feature/train_text.npy', train_text)
            '''
        with open('./training_nolabel.txt') as f:
            for l in f:
                test_text.append(l.strip())
                np.save('./feature/train_text.npy', train_text)
            '''
        test_text = [line.strip() for line in open(
            './training_nolabel.txt', "U", encoding='utf-8-sig')]
        np.save('./feature/test_text.npy', test_text)
    return y_train, train_text, test_text


def load_data(training_label, training_nolabel, testing):
    with open(training_label, "r", encoding='utf-8-sig') as f:
        y_train = []
        train_text = []
        test_text = []
        for l in f:
            y_train.append(l.strip().split("+++$+++")[0])
            train_text.append(l.strip().split("+++$+++")[1])
    valid_text = [line.strip() for line in open(
        training_nolabel, "r", encoding='utf-8-sig')]
    test_text = [line.strip().split(',', 1)[1] for line in open(
        testing, "r", encoding='utf-8-sig')][1::]
    return y_train, train_text, valid_text, test_text


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(
        input_seq, maxlen=NEW_MAX_SEQUENCE_LENGTH)
    predict_result = model.predict_classes(pad_input_seq)
    print(SentimentDict[predict_result[0][0]])


def savemodel(model, path):
    model.save_weights(path)
    print("Saved model to disk")


def buildmodel():
    model = Sequential()
    model.add(Embedding(output_dim=64,
                        input_dim=MAX_NB_WORDS,
                        input_length=MAX_SEQUENCE_LENGTH))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    model.summary()
    return model


'''
    y_train, train_text, valid_text, test_text = load_data(training_label,  training_nolabel,testing)
    np.save(train_text_path, train_text)
    np.save(y_train_path, y_train)
    np.save(valid_text_path, valid_text)
    np.save(test_text_path, test_text)
'''

train_text = np.load(train_text_path)
y_train = np.load(y_train_path)
valid_text = np.load(valid_text_path)
test_text = np.load(test_text_path)

token = Tokenizer(num_words=MAX_NB_WORDS,
                  filters='×ð!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'"'", split=" ")

token.fit_on_texts(np.concatenate([train_text, valid_text, test_text]))

'''
    Z = [text_to_word_sequence(R, filters='123456789qwertyuiopasdfghjklzxcvbnm!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=" ")     for R in valid_text]
    Q = []
    for i in range(len(Z)):
        Q = Q + Z[i]
        print(i)

    QQQ=text_to_word_sequence(str(QQ), filters='\"\'')
    QQQQ = text_to_word_sequence(str(QQ), filters="\",'")
    QQQQ.split()
'''


def preprocess(text):
    text_seq = token.texts_to_sequences(text)
    print(max(text_seq, key=len))
    texting = sequence.pad_sequences(text_seq, maxlen=MAX_SEQUENCE_LENGTH)
    return texting


x_train = preprocess(train_text)
x_test = preprocess(test_text)
x_valid = preprocess(valid_text)

ACCearlyStopping = EarlyStopping(
    monitor='val_acc', patience=50, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model = buildmodel()
train_history = model.fit(x_train, y_train, verbose=2,
                          batch_size=batchsize, epochs=num_epoch,
                          callbacks=[checkpoint, ACCearlyStopping], validation_split=0.1)


show_train_history(train_history, 'acc', 'val_acc')

show_train_history(train_history, 'loss', 'val_loss')
model = load_model(
    'C:/Users/user/Desktop/weight/weights-improvement-03-0.796.hdf5')
prediction = model.predict_classes(x_test, batch_size=5000)

text = open(predict_path, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(prediction)):
    s.writerow([i, prediction[i][0]])
text.close()

'''
    Start unsupervised learning step
'''
valid_predict = model.predict(x_valid, batch_size=10000)
good_incidence = []
good_incidence_y = []


def prob_to_one_hot(prob):
    if prob < 0.5:
        return(0)
    else:
        return(1)


x_train = preprocess(train_text)
x_test = preprocess(test_text)
x_valid = preprocess(valid_text)

for i in range(len(valid_predict)):
    if abs(valid_predict[i] - 0.5) > 0.45:
        good_incidence_y.append(prob_to_one_hot(valid_predict[i]))
        good_incidence.append(list(x_valid[i]))
        print(i)
np.save(good_incidence_path, good_incidence)
y_final = np.concatenate([good_incidence_y, y_train])
x_final = np.concatenate([good_incidence, x_train])


SemiSupervisedModel = buildmodel()
Finalpath = "c:/Users/user/Desktop/weight/Model-{epoch:02d}-{val_acc:.3f}.hdf5"
Finalcheckpoint = ModelCheckpoint(
    Finalpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

train_history = SemiSupervisedModel.fit(x_final, y_final, verbose=2,
                                        batch_size=batchsize, epochs=6,
                                        callbacks=[Finalcheckpoint, ACCearlyStopping], validation_split=0.1)

Finalpath = "c:/Users/user/Desktop/weight/Model-03-0.811.hdf5"
SemiSupervisedModel = load_model(
    "c:/Users/user/Desktop/weight/Model-03-0.811.hdf5")
prediction = SemiSupervisedModel.predict_classes(x_test, batch_size=10000)

text = open(predict_path, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(prediction)):
    s.writerow([i, prediction[i][0]])
text.close()
