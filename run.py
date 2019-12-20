#!/usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore')

from twts_preprocess import *
from ruby_python import *
from load_data import *
from ML_CNN_keras import *

'''
This run.py file allows to get the same predictions that were submitted as the best submission on AIcrowd.
It uses Ruby 2.0 preprocessing, pre-trained GloVe word embeddings for Twitter and a Convolutional Neural Network.
'''

# ==========================================================================================LOAD DATA===============================================================================

full_train_pos, full_train_neg, test = load_datasets(option='full')

# ==========================================================================================PREPROCESSING===============================================================================

run_train_preprocessing(full_train_pos, full_train_neg, pp='ruby')
run_test_preprocessing(test, pp='ruby')

# ==========================================================================================LOAD PREPROCESSED DATA===============================================================================

pos, neg, test = load_preprocessed_data(pp='ruby')

# ==========================================================================================EMBEDDINGS AND CNN===============================================================================

BATCH_SIZE = 64
DIM_EMB = 200
MAX_SEQUENCE_LENGTH = 50
NUM_EPOCHS = 5
TESTSIZE = 0.2

SUBMISSION_PATH = './Submissions/ruby_CNN_200d.txt'
VECTORS_PATH = './Data/glove_pretrained/glove.twitter.27B.200d.txt'

run_ruby_CNN(pos, neg, test, DIM_EMB, TESTSIZE, VECTORS_PATH, NUM_EPOCHS, BATCH_SIZE, './test19dec18h.txt')












