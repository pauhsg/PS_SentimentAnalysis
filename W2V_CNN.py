#!/usr/bin/env python3

from __future__ import division, print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten
import wandb
from wandb.keras import WandbCallback
import numpy as np
from keras.preprocessing import text
from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
import os
import collections
import re
import string

from W2V_utilities import *
from ML_sklearn import *


# load data
RESULT_POS_PATH = './Data/produced/pp_pos_otpl_nd.txt'
RESULT_NEG_PATH = './Data/produced/pp_neg_otpl_nd.txt'
RES_PATH = './Data/produced/pp_test_otpl.txt'

# load the data files = list with each line being a tweet
result_pos = open(RESULT_POS_PATH, "r").read().splitlines()
result_neg = open(RESULT_NEG_PATH, "r").read().splitlines()
test_set = open(RES_PATH, "r").read().splitlines()

# variable definition
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300
num_epochs = 3
batch_size = 34

# construct test and train dataframes
train_df = build_train_df_cnn(result_pos,result_neg)
test_df = build_test_df(test_set)

# split into test and train
data_train, data_test = train_test_split(train_df, test_size=0.20, random_state=42)
TRAINING_VOCAB = get_vocab(data_train)
TEST_VOCAB = get_vocab(data_test)

# word embedding

word2vec_path = './Data/google_pretrained/GoogleNews-vectors-negative300.bin.gz'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

training_embeddings = get_word2vec_embeddings(word2vec, data_train, generate_missing=True)

# tokenization and padding
tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(data_train["Tweet"].tolist())

training_sequences,train_word_index,train_cnn_data = get_sequences_idx_padding(tokenizer,data_train,MAX_SEQUENCE_LENGTH)

train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights.shape)

test_sequences,test_word_idx,test_cnn_data = get_sequences_idx_padding(tokenizer,data_test,MAX_SEQUENCE_LENGTH)

X_train = train_cnn_data
X_test = test_cnn_data

# CNN definition
def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=False)
    
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [2,3,4,5,6]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)


    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.1)(l_merge)  
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model

label_names = ['Pos','Neg']

y_train = data_train[label_names].values
y_test = data_test[label_names].values

model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                len(list(label_names)))

# CNN training
hist = model.fit(X_train, y_train, epochs=num_epochs, validation_split=0.2, shuffle=True, batch_size=batch_size)

# Test CNN
predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)
labels = [1, 0]

prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])
    
prediction_labels = [-1 if pred == 0 else 1 for pred in prediction_labels]

sum(data_test.CNN_Labels==prediction_labels)/len(prediction_labels)
print("is the predicted accuracy")

# test set

r_TEST_VOCAB = get_vocab(test_df) 
Test_sequences,Test_word_idx,Test_cnn = get_sequences_idx_padding(tokenizer,test_df,MAX_SEQUENCE_LENGTH)

r_y_pred = model.predict(Test_cnn, batch_size=1024, verbose=1)

labels = [1, 0]
r_prediction_labels=[]
for p in r_y_pred:
    r_prediction_labels.append(labels[np.argmax(p)])
    
r_prediction_labels = [-1 if pred == 0 else 1 for pred in prediction_labels]

test_id = test_df['Tweet_submission_id'].to_numpy()
create_csv_submission(test_id,r_prediction_labels, "./Submissions/W2V_CNN.csv")