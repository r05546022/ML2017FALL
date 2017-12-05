#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from keras.models import load_model
from keras.preprocessing.text import *
from gensim.models import Word2Vec

MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 260
batchsize = 2000

####################
'''
testing = 'c:/Users/user/Desktop/testing_data.txt'
predict_path = 'c:/Users/user/Desktop/result/predict.csv'
W2Vmodel_path = 'C:/Users/user/Desktop/skipgram_'
Model_path = 'C:/Users/user/Desktop/weight/weights-improvement-62-0.817.hdf5'
'''
testing = 'testing_data.txt'
predict_path = 'predict.csv'
W2Vmodel_path = 'skipgram_'
Model_path = 'hw4_model.hdf5'

####################


def Load_Data(testing, W2Vmodel_path, Model_path):
    test_text = [line.strip().split(',', 1)[1] for line in open(
            testing, "r", encoding='utf-8-sig')][1::]
    W2Vmodel = Word2Vec.load(W2Vmodel_path)
    model = load_model(Model_path)
    return test_text, W2Vmodel, model


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def write_result(prediction, predict_path=predict_path):
    text = open(predict_path, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(prediction)):
        s.writerow([i, prediction[i][0]])
    text.close()


def index_array(X, max_length):
    return np.concatenate([[word2idx.get('_PAD') if word2idx.get(x) is None else word2idx.get(x) for x in X],
                           np.zeros((max_length - len(X)))])


def set_training_vector():
    test_list = [text_to_word_sequence(
            s, filters='×ð!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=True, split=" ") for s in test_text]
    x_test = np.array(
            [index_array(x, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM) for x in test_list])
    return x_test


if __name__ == '__main__':
    test_text, W2Vmodel, model = Load_Data(testing, W2Vmodel_path, Model_path)
    word2idx = {"_PAD": 0}
    vocab_list = [(k, W2Vmodel.wv[k]) for k, v in W2Vmodel.wv.vocab.items()]
    embeddings_matrix = np.zeros((len(W2Vmodel.wv.vocab.items()) + 1, W2Vmodel.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = vocab_list[i][1]
    
    test_list = [text_to_word_sequence(s, filters='×ð!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=True, split=" ") for s
                 in test_text]
    x_test = np.array([index_array(x, MAX_SEQUENCE_LENGTH) for x in test_list])
    # Predict model
    prediction = model.predict_classes(x_test, batch_size=5000)
    write_result(prediction)
