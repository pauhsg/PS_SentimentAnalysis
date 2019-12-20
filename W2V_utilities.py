#!/usr/bin/env python3

#-----------------------------------------------------IMPORTS------------------------------------------------------#
import pandas as ps
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.svm import LinearSVC
from gensim import models
import warnings 
warnings.simplefilter('ignore')
from proj2_helpers import *
from get_embeddings_ML import *
from ML_sklearn import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize, WordNetLemmatizer

#-----------------------------------------------------FUNCTIONS------------------------------------------------------#

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def build_train_df(train_pos,train_neg):
    # create labels
    label_pos = [1] * len(train_pos)
    #create a df
    pos_df = pd.DataFrame(list(zip(label_pos, train_pos)),columns=["Sentiment","Tweet"]) 
    del label_pos
    # create labels
    label_neg = [-1] * len(train_neg)
    # create a df
    neg_df = pd.DataFrame(list(zip(label_neg, train_neg)),columns=["Sentiment","Tweet"]) #create a df
    del label_neg
    # regroup the dfs, ignore index in order to get new ones (->no duplicate)
    train_df = pd.concat([pos_df,neg_df],ignore_index=True) #regroup the dfs, ignore index in order to get new ones (->no duplicate)
    train_tokens = [word_tokenize(sen) for sen in train_df.Tweet] 
    train_df['tokens'] = train_tokens
    # shuffle the rows
    train_df = train_df.sample(frac=1)     
    return train_df

def build_train_df_cnn(train_pos,train_neg):
    # create labels
    label_pos = [1] * len(train_pos)
    #create a df
    pos_df = pd.DataFrame(list(zip(label_pos, train_pos)),columns=["Sentiment","Tweet"]) 
    del label_pos
    
    # create labels
    label_neg = [-1] * len(train_neg)
    # create a df
    neg_df = pd.DataFrame(list(zip(label_neg, train_neg)),columns=["Sentiment","Tweet"]) #create a df
    del label_neg
    
    # regroup the dfs, ignore index in order to get new ones (->no duplicate)
    train_df = pd.concat([pos_df,neg_df],ignore_index=True) #regroup the dfs, ignore index in order to get new ones (->no duplicate)
    
    train_tokens = [word_tokenize(sen) for sen in train_df.Tweet] 
    
    train_df['tokens'] = train_tokens
    
    CNNLabel = [0 if val == -1 else 1 for val in train_df.Sentiment.values]
    
    train_df.insert(2,"CNN_Labels",CNNLabel)
    
    # shuffle the rows
    train_df = train_df.sample(frac=1) 
    
    pos = []
    neg = []
    for l in train_df.CNN_Labels:
        if l == 0:
            pos.append(0)
            neg.append(1)
        elif l == 1:
            pos.append(1)
            neg.append(0)
            
    train_df['Pos']= pos
    train_df['Neg']= neg
    
    return train_df

def build_test_df(test_set):    
    test_ids = np.linspace(1,10000,10000, dtype=int)
    # create a df
    test_df = pd.DataFrame(list(zip(test_ids, test_set)), columns=["Tweet_submission_id","Tweet"]) 
    test_tokens = [word_tokenize(sen) for sen in test_df.Tweet] 
    test_df['tokens'] = test_tokens    
    return test_df

def get_vocab(dataframe):
    all_words = [word for tokens in dataframe["tokens"] for word in tokens]
    sentence_lengths = [len(tokens) for tokens in dataframe["tokens"]]
    VOCAB = sorted(list(set(all_words)))
    print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
    print("Max sentence length is %s" % max(sentence_lengths))
    return VOCAB

def get_sequences_idx_padding(tokenizer,dataframe,maximumlength):
    sequences = tokenizer.texts_to_sequences(dataframe["Tweet"].tolist())    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))    
    data = pad_sequences(sequences, maxlen=maximumlength)    
    return sequences,word_index,data

def scale_pca(train,test): 
    # create instance of StandardScaler
    scaler = StandardScaler()
    # fit on train set only
    scaler.fit(train)
    # apply transform to train and test
    X_train = scaler.transform(train)
    X_test = scaler.transform(test)
    # create instance of PCA
    pca = PCA(.95)
    # fit PCA on train set only
    pca.fit(X_train)
    # apply on train and test 
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train,X_test