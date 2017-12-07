#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Bidirectional ,Dropout
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import *
from gensim.models import word2vec, Word2Vec

MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 260
VALIDATION_SPLIT = 0.1

batch_size = 3000
num_epoch = 62
Min_count = 5
####################
training_label = sys.argv[1]    
#training_label =  'c:/Users/user/Desktop/training_label.txt'
training_nolabel = sys.argv[2]  
#training_nolabel =  'c:/Users/user/Desktop/training_nolabel.txt'

W2Vmodel_path = 'skipgram_'
Model_path = 'hw4_model.hdf5'


####################


def load_data(training_label, training_nolabel):
    with open(training_label, "r", encoding='utf-8-sig') as f:
        y_train = []
        train_text = []
        test_text = []
        for l in f:
            y_train.append(l.strip().split("+++$+++")[0])
            train_text.append(l.strip().split("+++$+++")[1])
    valid_text = [line.strip() for line in open(training_nolabel, "r", encoding='utf-8-sig')]
    return y_train, train_text, valid_text


def buildmodel():
    model = Sequential()
    model.add(Embedding(len(embeddings_matrix), EMBEDDING_DIM,
                        weights=[embeddings_matrix], trainable=False))
    model.add(LSTM(256))
    model.add(Dense(units=1, activation='relu'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    model.summary()
    return model



def Set_W2V_model():
    documents = np.concatenate([train_text, valid_text])
    sentences = [text_to_word_sequence(s, filters='', lower=True, split=" ") for s in
                 documents.tolist()]
    W2Vmodel = Word2Vec(sentences, sg=1, size=EMBEDDING_DIM,
                        window=5, min_count=Min_count, workers=8)
    W2Vmodel.save(W2Vmodel_path)
    
    return W2Vmodel


def index_array(X, max_length, EMBEDDING_DIM):
    return np.concatenate([[word2idx.get('_PAD') if word2idx.get(x) is None else word2idx.get(x) for x in X],
                           np.zeros((max_length - len(X)))])



def set_training_vector():
    train_list = [text_to_word_sequence(s, filters='', lower=True, split=" ") for s in train_text]
    valid_list = [text_to_word_sequence(s, filters='', lower=True, split=" ") for s in valid_text]
    x_train = np.array( [index_array(x, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM) for x in train_list])
    x_valid = np.array([index_array(x[:(MAX_SEQUENCE_LENGTH - 1)], MAX_SEQUENCE_LENGTH, EMBEDDING_DIM) for x in valid_list])
    return x_train,  x_valid


y_train, train_text, valid_text = load_data(training_label, training_nolabel)
W2Vmodel = Set_W2V_model()
W2Vmodel = Word2Vec.load(W2Vmodel_path)
word2idx = {"_PAD": 0}
vocab_list = [(k, W2Vmodel.wv[k]) for k, v in W2Vmodel.wv.vocab.items()]
embeddings_matrix = np.zeros((len(W2Vmodel.wv.vocab.items()) + 1, W2Vmodel.vector_size))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 1
    embeddings_matrix[i + 1] = vocab_list[i][1]
x_train, x_valid = set_training_vector()
ACCearlyStopping = EarlyStopping(
        monitor='val_acc', patience=50, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(
    Model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model = buildmodel()
train_history = model.fit(x_train, y_train, verbose=2,
                          batch_size=batch_size, epochs=num_epoch,
                          callbacks=[checkpoint, ACCearlyStopping], validation_split=VALIDATION_SPLIT,
                          class_weight='auto')
