#!/usr/bin/env python3
import numpy as np
import pandas as pd

def get_vectors(vectors_filepath):
    '''
    get a dictionary with {'word': vector}
    '''
    vectors = {} 
    with open(vectors_filepath, "r") as f:
        for line in f:
            tokens = line.strip().split()
            vectors[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    vectors 
    return vectors

def compute_mean_we_glove(data, vectors, dim_emb):
    '''
    computes mean of word vectors for each tweet
    '''
    print('> computing mean of word vectors')
    
    # word vectors of data 
    word_vectors = [[vectors.get(t, np.zeros(dim_emb)) for t in line.strip().split()] for line in data]

    # tweet vectors -> mean of word vectors for each tweet
    twt_vectors = [np.mean(word_vectors[i], axis=0) for i in range(len(word_vectors))]   

    return twt_vectors

def get_train_emb(pos, neg, vectors, dim_emb):
    '''
    given the preprocessed positive and negative tweets data, the vectors and the embedding dimension, 
    extracts mean of word vectors per tweets, and outputs a dataframe containing all pos 
    and neg mean word vectors and their labels (1 for pos/ -1 for neg), then shuffles 
    the rows and also outputs the X matrix containing mean word vectors and the vector y containing
    the labels, ready to be used into ML algorithms
    '''
    print('> process pos and neg datas to get X and y to perform ML')

    # compute mean word embeddings of each tweets for pos
    twt_vectors_pos = compute_mean_we_glove(pos, vectors, dim_emb)

    # labels 1 for positive tweets + create dataFrame with mean word emb
    label_pos = [1] * len(twt_vectors_pos)
    df_pos = pd.DataFrame(list(zip(label_pos, twt_vectors_pos)), columns=["sentiment", "twt_vec"])
    del label_pos

    # compute mean word embeddings of each tweets for neg
    twt_vectors_neg = compute_mean_we_glove(neg, vectors, dim_emb)

    # labels -1 for negative tweets + create dataFrame with mean word emb
    label_neg = [-1] * len(twt_vectors_neg)
    df_neg = pd.DataFrame(list(zip(label_neg, twt_vectors_neg)), columns=["sentiment", "twt_vec"])
    del label_neg

    # drop NaN
    df_pos.dropna(inplace = True)
    df_neg.dropna(inplace = True)

    # regroup the dfs, ignore index in order to get new ones (->no duplicate)
    full_df = pd.concat([df_pos, df_neg],ignore_index=True)
    # shuffles the rows
    full_df = full_df.sample(frac=1) 

    # create X
    mat = full_df['twt_vec'].to_numpy()
    X = np.concatenate(mat, axis=0).reshape((mat.shape[0], dim_emb))

    # create y
    y = full_df['sentiment'].to_numpy()

    print('> X and y informations:')
    print('X shape:', X.shape)
    print('y shape:', y.shape)

    return full_df, X, y


def get_test_emb(test, vectors, dim_emb):
    '''
    given the preprocessed test set, the vectors and the embedding dimension, extracts mean of word 
    vectors per tweets, and outputs a dataframe containing all tweets mean word vectors
    and their labels (1 for pos/ -1 for neg) and also outputs the testx 
    matrix containing mean word vectors ready to be put in ML algorithms
    '''
    print('> process test data to get X_test and perform ML')

    # compute mean word embeddings of each tweets for test
    twt_vectors_test = compute_mean_we_glove(test, vectors, dim_emb)

    # create test ids
    test_ids = np.linspace(1,10000,10000, dtype=int)

    # create dataFrame 
    df_test = pd.DataFrame(list(zip(test_ids, twt_vectors_test)), columns=["Tweet_submission_id", "twt_vec"])
    del test_ids

    testx = df_test['twt_vec'].to_numpy()
    testx = [x if not np.isnan(x).any() else np.zeros((dim_emb,)) for x in testx]
    testx = np.concatenate(testx, axis=0).reshape((df_test.shape[0], dim_emb))

    print('Test shape', testx.shape)

    return df_test, testx

