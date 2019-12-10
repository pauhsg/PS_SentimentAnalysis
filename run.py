#!/usr/bin/env python3
from twt_preprocess import *

DATA_TRAIN_POS_PATH = './Datasets/twitter-datasets/pos_train.txt'  
DATA_TRAIN_NEG_PATH = './Datasets/twitter-datasets/neg_train.txt'

pos_data = open(DATA_TRAIN_POS_PATH, "r").read().splitlines()
neg_data = open(DATA_TRAIN_NEG_PATH, "r").read().splitlines()

pos_data_ = pos_data[20:35]

essai = preprocessing(pos_data_)
save_preprocessed_data(essai, ['o.txt','a.txt','e.txt','w.txt'], True)
save_preprocessed_data(essai, ['o.txt','a.txt','e.txt','w.txt'])

