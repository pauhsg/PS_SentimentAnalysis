#!/usr/bin/env python3
import numpy as np
import pandas as pd
import re
regex = re.compile('[^A-Za-zÀ-ÿ]')

def extract_mean_word_vectors(data, vocabulary, embeddings):
    print('> extracting mean of word vectors')
    
    # get vocab equivalence to tweet words
    idx_data = [[vocabulary.get((regex.sub(' ', ' '.join(regex.sub(' ', t).split()))), -1) for t in line.strip().split()] for line in data]
    idx_data = [[t for t in tokens if t>=0] for tokens in idx_data]
    
    # get dense vector equivalence to tweet words
    data_tweets_word_vector = [[embeddings[wd2voc][:] for wd2voc in tweet_words] for tweet_words in idx_data]

    # get mean word vector of each tweet
    data_tweets_mean_vector = [np.mean(wordvectors,axis=0) for wordvectors in data_tweets_word_vector]
    
    return idx_data, data_tweets_word_vector, data_tweets_mean_vector

def process_train_ML(pos, neg, vocabulary, embeddings, dim_emb):
    print('> process pos and neg datas to get X and y to perform ML')
    
    # seperate list of tweets in lines
    #pos = [x.strip() for x in pos[0].split(',')]
    #neg = [x.strip() for x in neg[0].split(',')]
    
    # extract mean word embeddings
    idx_pos_tweets, pos_tweets_word_vector, pos_tweets_mean_vector = extract_mean_word_vectors(pos, vocabulary, embeddings)
    idx_neg_tweets, neg_tweets_word_vector, neg_tweets_mean_vector = extract_mean_word_vectors(neg, vocabulary, embeddings)
    
    # create labels
    label_pos = [1] * len(pos)
    #create a df
    pos_df = pd.DataFrame(list(zip(label_pos, pos, idx_pos_tweets, pos_tweets_word_vector, pos_tweets_mean_vector)),\
                      columns=["Sentiment","Tweet","Token_idx","Words_Vectors","Mean_Word_Vector"]) 
    del label_pos
    
    # create labels
    label_neg = [-1] * len(neg)
    # create a df
    neg_df = pd.DataFrame(list(zip(label_neg, neg, idx_neg_tweets, neg_tweets_word_vector, neg_tweets_mean_vector)),\
                      columns=["Sentiment","Tweet","Token_idx","Words_Vectors","Mean_Word_Vector"]) #create a df
    del label_neg
    
    # regroup the dfs, ignore index in order to get new ones (->no duplicate)
    full_df = pd.concat([pos_df,neg_df],ignore_index=True) #regroup the dfs, ignore index in order to get new ones (->no duplicate)

    # shuffle the rows
    full_df = full_df.sample(frac=1) 
    
    print('> X and y informations:')
    # get X matrix
    X = full_df['Mean_Word_Vector'].to_numpy()
    X = [x if not np.isnan(x).any() else np.zeros((20,)) for x in X]
    X = np.concatenate(X, axis=0).reshape((full_df.shape[0], dim_emb))
    print('X shape:', X.shape)
    
    # get y
    y = full_df['Sentiment'].to_numpy()
    print('y shape:', y.shape)
    
    return full_df, X, y

def process_test_ML(test, vocabulary, embeddings, dim_emb):
    print('> process test data to get X_test and perform ML')
    
    # extract mean word embeddings
    idx_test_tweets,test_tweets_word_vector,test_tweets_mean_vector = extract_mean_word_vectors(test, vocabulary, embeddings)
    
    # create labels
    test_ids = np.linspace(1,10000,10000, dtype=int)
    # create a df
    test_df = pd.DataFrame(list(zip(test_ids, test, idx_test_tweets,test_tweets_word_vector,test_tweets_mean_vector)),\
                      columns=["Tweet_submission_id","Tweet","Token_idx","Words_Vectors","Mean_Word_Vector"]) 
    del test_ids
    
    print('> X_test informations:')
    # get X_test matrix
    X_test = test_df['Mean_Word_Vector'].to_numpy()
    X_test = [x if not np.isnan(x).any() else np.zeros((20,)) for x in X_test]
    X_test = np.concatenate(X_test, axis=0).reshape((test_df.shape[0], dim_emb))
    print('X_test shape:', X_test.shape)
    
    return test_df, X_test
    
    

    
    





