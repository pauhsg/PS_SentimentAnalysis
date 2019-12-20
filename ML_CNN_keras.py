#!/usr/bin/env python3
from __future__ import division, print_function

import numpy as np
import pandas as pd 

from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

from get_GloVe_emb_ML import *
from proj2_helpers import *

MAX_SEQUENCE_LENGTH = 50

def get_train_df_CNN(pos, neg):
    '''
    given the preprocessed positive and negative tweets, creates a dataframe containing all pos 
    and neg tweets and their sentiment (1 for pos/ -1 for neg), then shuffles the rows and outputs it
    '''
    print('> create a Pandas DataFrame with preprocessed and shuffled pos and neg tweets to perform CNN')

    # labels 1 for positive tweets + create dataFrame with mean word emb
    label_pos = [1] * len(pos)
    df_pos = pd.DataFrame(list(zip(label_pos, pos)), columns=['sentiment', 'twt'])
    del label_pos

    # labels -1 for negative tweets + create dataFrame with mean word emb
    label_neg = [-1] * len(neg)
    df_neg = pd.DataFrame(list(zip(label_neg, neg)), columns=['sentiment', 'twt'])
    del label_neg

    # drop NaN
    df_pos.dropna(inplace = True)
    df_neg.dropna(inplace = True)

    # regroup the dfs, ignore index in order to get new ones (->no duplicate)
    full_df = pd.concat([df_pos, df_neg], ignore_index=True)

    # shuffles the rows
    full_df = full_df.sample(frac=1) 

    print('full_df shape: ', full_df.shape)

    return full_df

def get_test_df_CNN(test):
    '''
    given the preprocessed test tweets, creates a dataframe containing all tweets and their id
    and outputs it
    '''
    print('> create a Pandas DataFrame with preprocessed test tweets to perform CNN')

    # create test ids
    test_ids = np.linspace(1,10000,10000, dtype=int)

    # create dataFrame 
    df_test = pd.DataFrame(list(zip(test_ids, test)), columns=['Tweet_submission_id', 'twt'])
    del test_ids
    
    print('df_test shape: ', df_test.shape)

    return df_test


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    '''
    Convolutional Neural Network from https://github.com/saadarshad102/Sentiment-Analysis-CNN
    '''
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
        l_conv = Conv1D(filters=200, 
                        kernel_size=filter_size, 
                        activation='relu')(embedded_sequences)
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


def train_ruby_CNN(pos, neg, dim_emb, testsize, vectors_path, num_epochs, batch_size):
    '''
    given preprocessed pos, neg and test data, the embedding dimension the vectors' file path 
    and the test size, runs the a convolutional neural netword (CNN) to predict if tweets 
    are positive or negative!

    adapted from https://github.com/saadarshad102/Sentiment-Analysis-CNN

    '''
    print('> preparing data and training CNN with an embedding dimension of', dim_emb, 'and a test size of', testsize)

    # get train DataFrame
    data = get_train_df_CNN(pos, neg)

    # tokenize keeping our tags like <user> in a single token and store them in a new column of the DataFram
    tokens = [sen.split() for sen in data.twt]
    data['tokens'] = tokens

    # transform labels into one hot encoded columns 
    pos_lab = []
    neg_lab = []
    for l in data.sentiment:
        if l == -1:
            pos_lab.append(0)
            neg_lab.append(1)
        elif l == 1:
            pos_lab.append(1)
            neg_lab.append(0)
    data['Pos']= pos_lab
    data['Neg']= neg_lab
    data = data[['twt', 'tokens', 'sentiment', 'Pos', 'Neg']]
    
    # split data into train and test
    data_train, data_test = train_test_split(data, test_size=testsize, random_state=42)

    # build training vocabulary
    all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
    TRAINING_VOCAB = sorted(list(set(all_training_words)))

    # load GloVe pre-trained word embeddings
    print('> loading GloVe pre-trained word embeddings (this step can take a while)')
    glove2word2vec(glove_input_file=vectors_path, word2vec_output_file="./Data/produced/gensim_glove_vectors.txt")
    glove_model = KeyedVectors.load_word2vec_format("./Data/produced/gensim_glove_vectors.txt", binary=False)

    # train tokenizer on train, tokenize and pad sequences
    tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
    tokenizer.fit_on_texts(data_train['twt'].tolist())
    training_sequences = tokenizer.texts_to_sequences(data_train['twt'].tolist())
    train_word_index = tokenizer.word_index

    train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    train_embedding_weights = np.zeros((len(train_word_index)+1, dim_emb))
    for word,index in train_word_index.items():
        train_embedding_weights[index,:] = glove_model[word] if word in glove_model else np.random.rand(dim_emb)

    test_sequences = tokenizer.texts_to_sequences(data_test['twt'].tolist())
    test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # get labels 
    label_names = ['Pos', 'Neg']
    y_train = data_train[label_names].values

    # initialise model
    print('> Model summary: ')
    model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, dim_emb, len(list(label_names)))
    
    # train model
    print('> Training CNN')
    hist = model.fit(train_cnn_data, y_train, epochs=num_epochs, validation_split=0.2, shuffle=True, batch_size=batch_size)

    # test model
    print('> Testing CNN')
    predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)
    labels = [1, 0]
    prediction_labels=[]
    for p in predictions:
        prediction_labels.append(labels[np.argmax(p)])

    # convert 0, 1 labels into -1, 1 labels
    prediction_labels=[-1 if pred == 0 else 1 for pred in prediction_labels]

    # compute test accuracy
    sum(data_test.sentiment==prediction_labels)/len(prediction_labels)
    print('Obtained accuracy on test: ', sum(data_test.sentiment==prediction_labels)/len(prediction_labels))

    return model, tokenizer


def run_ruby_CNN(pos, neg, test, dim_emb, testsize, vectors_path, num_epochs, batch_size, submission_path):
    '''
    given all needed data, will perform training of CNN and then apply it to the test set
    and save a submission in Submissions folder using submission_path

    adapted from https://github.com/saadarshad102/Sentiment-Analysis-CNN
    
    '''
    print('>> RUNNING CNN ')
    # get test DataFrame
    df_test = get_test_df_CNN(test)

    # tokenize 
    tokens = [sen.split() for sen in df_test.twt]
    df_test['tokens'] = tokens

    model, tokenizer = train_ruby_CNN(pos, neg, dim_emb, testsize, vectors_path, num_epochs, batch_size)

    # tokenize using trained tokenizer and pad 
    test_sequences_TEST = tokenizer.texts_to_sequences(df_test['twt'].tolist())
    test_cnn_data_TEST = pad_sequences(test_sequences_TEST, maxlen=MAX_SEQUENCE_LENGTH)

    # make prediction on the test set
    predictions_TEST = model.predict(test_cnn_data_TEST, batch_size=1024, verbose=1)

    # get labels
    labels_TEST = [1, 0]
    prediction_labels_TEST = []
    for p in predictions_TEST:
        prediction_labels_TEST.append(labels_TEST[np.argmax(p)])

    # transform 0, 1 labels into -1, 1
    prediction_labels_TEST = [-1 if pred == 0 else 1 for pred in prediction_labels_TEST]

    # create and save submission using submission_path
    create_submission(df_test, prediction_labels_TEST, submission_path)
    print('---> submission ready in Submissions folder')





    
    
    
    
    
    
    










