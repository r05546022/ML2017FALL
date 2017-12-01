import numpy as np

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import os
import tempfile
TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

from gensim import corpora


train_text_path = 'c:/Users/user/Desktop/feature/train_text.npy'
valid_text_path = 'c:/Users/user/Desktop/feature/valid_text.npy'
test_text_path = 'c:/Users/user/Desktop/feature/test_text.npy'
train_text = np.load(train_text_path)
valid_text = np.load(valid_text_path)
test_text = np.load(test_text_path)
texts = np.concatenate([train_text, valid_text, test_text])
