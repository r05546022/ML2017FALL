import sys
import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import keras
from keras.layers import Dense, Input
from keras.models import Model
import numpy as np
from keras.models import load_model

'''
image_path = 'C:/Users/user/Desktop/image.npy'
test_case_path = 'C:/Users/user/Desktop/test_case.csv'
prediction_path = 'C:/Users/user/Desktop/prediction.csv'
'''
image_path = sys.argv[1]
test_case_path = sys.argv[2]
prediction_path = sys.argv[3]
autoencoder_model_path = 'hw6_autoencoder.h5'
encoder_model_path = 'hw6_encoder.h5'

EPOCHS = 100
BATCH_SIZE = 5000
VALID_SPLIT = 0.1


image = np.load(image_path)
test_case = pd.read_csv(test_case_path)

def write_result(prediction, prediction_path=prediction_path):
    text = open(prediction_path, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["ID", "ANS"])
    for i in range(len(prediction)):
        s.writerow([i, prediction[i]])
    text.close()

def evaluate_testing(cluster_labels):
    I1 = [cluster_labels[i] for i in test_case['image1_index']]
    I2 = [cluster_labels[i] for i in test_case['image2_index']]
    result = []
    for i in range(len(I1)):
        result.append(1 * (I1[i] == I2[i]))
    write_result(result)

def build_model():
    
    input_image = Input(shape=(28 * 28,), name='InputImage')
    encoded = Dense(512, activation='relu')(input_image)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoder_output = Dense(28 * 28)(encoded)

    decoded = Dense(64, activation='relu')(encoder_output)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    output_image = Dense(28 * 28, activation='tanh')(decoded)

    # construct the autoencoder model
    autoencoder = Model(input=input_image, output=output_image)
    # construct the encoder model for plotting
    encoder = Model(input=input_image, output=encoder_output)
    encoder.summary()
    autoencoder.summary()
    return autoencoder, encoder

autoencoder, encoder = build_model()
'''
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(image / 255. - 0.5,
                image / 255. - 0.5,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True, validation_split=VALID_SPLIT)
autoencoder.save(autoencoder_model_path)
encoder.save(encoder_model_path)
'''

autoencoder = load_model(autoencoder_model_path)
encoder = load_model(encoder_model_path)

encoder_images = encoder.predict(image / 255. - 0.5)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoder_images)
evaluate_testing(kmeans.labels_)


"""


def meanX(dataX):
    return np.mean(dataX, axis=0)

def pca(XMat, k):
    average = meanX(XMat)
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T) 
    featValue, featVec = np.linalg.eig(covX)  
    index = np.argsort(-featValue)
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        # 注意特征向量时列向量，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]])
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData

finalData, reconData = pca(image, 64)

FinalData = np.array(finalData)
#FinalData = np.array(np.transpose(finalData))

sc = plt.scatter(np.array(np.transpose(finalData))[0], np.array(np.transpose(finalData))[1])
plt.show()

kmeans = KMeans(n_clusters=2, random_state=0).fit(FinalData)
evaluate_testing(kmeans.labels_)

for i in range(61,65):
    II = np.reshape(reconData[i], (28, 28))
    imgplot = plt.imshow(II)
    plt.show()

"""
