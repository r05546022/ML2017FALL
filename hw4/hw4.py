#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, TimeDistributed, Bidirectional
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import *
from gensim.models import word2vec, Word2Vec

MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 240
VALIDATION_SPLIT = 0.2

batchsize = 1024
num_epoch = 20

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
W2Vmodel_path = 'C:/Users/user/Desktop/skipgram_'


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


def savemodel(model, path):
    model.save_weights(path)
    print("Saved model to disk")


def buildmodel():
    model = Sequential()
    model.add(Embedding(len(embeddings_matrix), EMBEDDING_DIM,
                        weights=[embeddings_matrix], trainable=False))
    # model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    model.summary()
    return model


def Set_Data(readnpy=False):
    if readnpy:
        y_train, train_text, valid_text, test_text = load_data(
                training_label, training_nolabel, testing)
        np.save(train_text_path, train_text)
        np.save(y_train_path, y_train)
        np.save(valid_text_path, valid_text)
        np.save(test_text_path, test_text)
    else:
        train_text = np.load(train_text_path)
        y_train = np.load(y_train_path)
        valid_text = np.load(valid_text_path)
        test_text = np.load(test_text_path)
    return train_text, y_train, valid_text, test_text


def Set_W2V_model():
    documents = np.concatenate([train_text, valid_text, test_text])
    sentences = [text_to_word_sequence(s, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=True, split=" ") for s in
                 documents.tolist()]
    W2Vmodel = Word2Vec(sentences, sg=1, size=EMBEDDING_DIM,
                        window=5, min_count=5, workers=8)
    W2Vmodel.save(W2Vmodel_path)


def index_array(X, max_length, EMBEDDING_DIM):
    return np.concatenate([[word2idx.get('_PAD') if word2idx.get(x) is None else word2idx.get(x) for x in X],
                           np.zeros((max_length - len(X)))])


def write_result(prediction=prediction, predict_path=predict_path):
    text = open(predict_path, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(prediction)):
        s.writerow([i, prediction[i][0]])
    text.close()


train_text, y_train, valid_text, test_text = Set_Data()
Set_W2V_model()

W2Vmodel = Word2Vec.load(W2Vmodel_path)
word2idx = {"_PAD": 0}
vocab_list = [(k, W2Vmodel.wv[k]) for k, v in W2Vmodel.wv.vocab.items()]
embeddings_matrix = np.zeros(
        (len(W2Vmodel.wv.vocab.items()) + 1, W2Vmodel.vector_size))

for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 1
    embeddings_matrix[i + 1] = vocab_list[i][1]

train_list = [text_to_word_sequence(s, filters='×ð!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=True, split=" ") for s in
              train_text]
test_list = [text_to_word_sequence(s, filters='×ð!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=True, split=" ") for s in
             test_text]
valid_list = [text_to_word_sequence(s, filters='×ð!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=True, split=" ") for s in
              valid_text]

train_vec = np.array(
        [index_array(x, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM) for x in train_list])
test_vec = np.array(
        [index_array(x, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM) for x in test_list])
valid_vec = np.array([index_array(x[:(MAX_SEQUENCE_LENGTH - 1)],
                                  MAX_SEQUENCE_LENGTH, EMBEDDING_DIM) for x in valid_list])

ACCearlyStopping = EarlyStopping(
        monitor='val_acc', patience=50, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model = buildmodel()
train_history = model.fit(train_vec, y_train, verbose=2,
                          batch_size=2000, epochs=num_epoch,
                          callbacks=[checkpoint, ACCearlyStopping], validation_split=VALIDATION_SPLIT,
                          class_weight='auto')

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

# Predict model
model = load_model(
        'c:/Users/user/Desktop/weight/weights-improvement-12-0.814.hdf5')
prediction = model.predict_classes(test_vec, batch_size=5000)
write_result(prediction)
