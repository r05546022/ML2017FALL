#!/bin/bash 
wget -O hw4_model.hdf5        'https://www.dropbox.com/s/g7rorlciykpbobq/hw4_model.hdf5?dl=1'
wget -O skipgram_.wv.syn0.npy 'https://www.dropbox.com/s/lkdapcwgy3wcqqf/skipgram_.wv.syn0.npy?dl=1'
wget -O skipgram_.syn1neg.npy 'https://www.dropbox.com/s/kj3p3e4yw4yml0n/skipgram_.syn1neg.npy?dl=1'
wget -O skipgram_             'https://www.dropbox.com/s/6nfpuvxgoxvmnij/skipgram_?dl=1'
python hw4_train.py $1 $2