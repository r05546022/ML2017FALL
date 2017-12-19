
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Bidirectional ,Merge ,Input, Dot, Reshape, merge, Dropout,Add
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, sgd
from keras.preprocessing import sequence

from keras.layers import *
from numpy import bincount, ravel, log
from keras import backend as K
from keras.models import Model, Sequential
from sklearn.utils import shuffle


train_path = 'C:/Users/user/Desktop/train.csv'
test_path = 'C:/Users/user/Desktop/test.csv'
movies_path = 'C:/Users/user/Desktop/movies.csv'
users_path = 'C:/Users/user/Desktop/users.csv'
filepath =  'C:/Users/user/Desktop/weight/weight'
predict_path =  'C:/Users/user/Desktop/PRED.csv'

'''
train_path = 'train.csv'
test_path = 'test.csv'
movies_path = 'movies.csv'
users_path = 'users.csv'
'''

def write_result(prediction, predict_path=predict_path):
    text = open(predict_path, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["TestDataID", "Rating"])
    for i in range(len(prediction)):
        s.writerow([i+1, abs(prediction[i])])
    text.close()


train  = shuffle(pd.read_csv(train_path))
test   = pd.read_csv(test_path)
movies = pd.read_csv(movies_path, delimiter='::')
users  = shuffle(pd.read_csv(users_path))

RATINGS_CSV_FILE = 'Prediction.csv'
MODEL_WEIGHTS_FILE = 'ml1m_weights.h5'
VALIDATION_SPLIT = 0.15

K_FACTORS = 120
max_userid = train['UserID'].drop_duplicates().max()
max_movieid = train['MovieID'].drop_duplicates().max()


def MF_model(n_users, n_items, latent_dim = K_FACTORS):
    user_input = Input(shape = [1], name='UserID')
    item_input = Input(shape = [1], name='MovieID')
    user_vec = Embedding(n_users+1,latent_dim)(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items+1,latent_dim)(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users+1,1)(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items+1,1)(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes = 1)([user_vec,item_vec])
    r_hat = Add()([r_hat,user_bias,item_bias])
    model = keras.models.Model([user_input,item_input],r_hat)
    model.compile(loss = 'mse',optimizer = 'Adam')
    return(model)


def nn_model(n_users, n_items, latent_dim = 6666):
    user_input = Input(shape = [1])
    item_input = Input(shape = [1])
    user_vec = Embedding(n_users,latent_dim,embeddings_initializer = 'random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items,latent_dim,embeddings_initializer = 'random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    merge_vec = Concatenate()([user_vec,item_vec])
    hidden = Dense(150,activation = 'relu')(merge_vec)
    hidden = Dense(50,activation = 'relu')(hidden)
    output = Dense(1)(hidden)
    model = keras.models.Model([user_input,item_input],output)
    model.compile(loss = 'mse',optimizer = 'sgd')
    model.summary()
    return(model)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

ACCearlyStopping = EarlyStopping(
        monitor='val_mse', patience=50, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

MFModel = MF_model(max_userid,max_movieid)
train_history = MFModel.fit([train['UserID'], train['MovieID']], train['Rating'], verbose=1,batch_size=2000, epochs=10,callbacks=[checkpoint, ACCearlyStopping], validation_split=0.05,class_weight='auto')


show_train_history(train_history, 'loss', 'val_loss')

ANS=MFModel.predict([test['UserID'], test['MovieID']])
ANS2 = np.transpose(ANS)[0]
write_result(ANS2)

nnModel = nn_model(max_userid,max_movieid)
train_history = nnModel.fit([train['UserID'], train['MovieID']], train['Rating'], verbose=1,batch_size=2000, epochs=10,callbacks=[checkpoint, ACCearlyStopping], validation_split=0.1,class_weight='auto')


#get embedding
user_emb = np.array(train_history.model.layers[2].get_weights()).squeeze()
print('user embedding shape:', user_emb.shape)
movie_emb = np.array(train_history.model.layers[3].get_weights()).squeeze()
print('user embedding shape:', movie_emb.shape)
np.save('user_emb.npy',user_emb)
np.save('movie_emb.npy',movie_emb)

# Draw t-sne
List = pd.DataFrame( list(range(3953)))
List = List.rename(columns={0: 'movieID'})
List = pd.merge(List, movies, how='left', on='movieID', sort=False)['Genres']
List_2_str = [str(item) for item in List]
LIST = [L.split("|")[0] for L in List_2_str]
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
y = np.array(pd.factorize(LIST)[0])
x = np.array(movie_emb,dtype=np.float64)
#perform t-SNE embedding
vis_data = TSNE(n_components=2).fit_transform(x)
#plot the result
vis_x = vis_data[:,0]
vis_y = vis_data[:,1]
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(vis_x,vis_y,c=y,cmap=cm)
plt.colorbar(sc)
plt.show()

def nn_model(n_users, n_items, latent_dim = 6666):
    user_input = Input(shape = [1])
    item_input = Input(shape = [1])
    user_vec = Embedding(n_users,latent_dim)(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items,latent_dim)(item_input)
    item_vec = Flatten()(item_vec)
    merge_vec = Concatenate()([user_vec,item_vec])
    hidden = Dropout(0.2)(merge_vec)
    hidden = Dense(128,activation = 'relu',kernel_regularizer=regularizers.l2(0.001))(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Dropout(0.4)(hidden)
    hidden = Dense(64,activation = 'relu',kernel_regularizer=regularizers.l2(0.001))(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Dropout(0.4)(hidden)
    output = Dense(32,activation='linear')(hidden)  
    model = keras.models.Model([user_input,item_input],output)
    model.compile(loss = 'mse',optimizer = 'adam')
    model.summary()
    return(model)
