import sys
import csv
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.layers import *


'''
test_path = 'C:/Users/user/Desktop/test.csv'
predict_path = 'C:/Users/user/Desktop/PREDICT.csv'
'''
test_path = sys.argv[1]
predict_path = sys.argv[2]
movies_path = sys.argv[3]
users_path = sys.argv[4]

test = pd.read_csv(test_path)

def write_result(prediction, predict_path=predict_path):
    text = open(predict_path, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["TestDataID", "Rating"])
    for i in range(len(prediction)):
        s.writerow([i + 1, abs(prediction[i])])
    text.close()


MODEL = load_model('hw5_MFmodel.h5')
ANS = MODEL.predict([test['UserID'], test['MovieID']])
write_result(np.transpose(ANS)[0])
