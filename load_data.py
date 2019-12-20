#!/usr/bin/env python3

def load_datasets(option):
    '''
    loads datasets available in the twitter-datasets folder:
        - if option = 'subset', the function will load the subsets of 100 000 tweets and not the full sets
        - if option = 'full', the function will load the full tweets sets
    '''
    if option == 'subset':
        TRAIN_POS_PATH = './Data/twitter-datasets/train_pos.txt'  
        TRAIN_NEG_PATH = './Data/twitter-datasets/train_neg.txt'
        TEST_PATH = './Data/twitter-datasets/test_data.txt'

        train_pos = open(TRAIN_POS_PATH, "r").read().splitlines()
        train_neg = open(TRAIN_NEG_PATH, "r").read().splitlines()
        test = open(TEST_PATH, "r").read().splitlines()

        return train_pos, train_neg, test


    if option == 'full':
        FULL_TRAIN_POS_PATH = './Data/twitter-datasets/train_pos_full.txt'  
        FULL_TRAIN_NEG_PATH = './Data/twitter-datasets/train_neg_full.txt'
        TEST_PATH = './Data/twitter-datasets/test_data.txt'

        # load the data files = list with each line being a tweet
        full_train_pos = open(FULL_TRAIN_POS_PATH, "r").read().splitlines()
        full_train_neg = open(FULL_TRAIN_NEG_PATH, "r").read().splitlines()
        test = open(TEST_PATH, "r").read().splitlines()

        return full_train_pos, full_train_neg, test

def load_preprocessed_data(pp):
    '''
    load data that has been preprocessed with the specified preprocessing (pp)  
    '''
    if pp == 'ruby':
        print('> loading Ruby 2.0 preprocessed data in the format one tweet per line')
        RUBY_FPOS_PATH = './Data/preprocessed/ruby_pos_otpl.txt' 
        RUBY_FNEG_PATH = './Data/preprocessed/ruby_neg_otpl.txt' 
        RUBY_TEST_PATH = './Data/preprocessed/ruby_test_otpl.txt'

        # load the data files = list with each line being a tweet
        pos = open(RUBY_FPOS_PATH, "r").read().splitlines()
        neg = open(RUBY_FNEG_PATH, "r").read().splitlines()
        test = open(RUBY_TEST_PATH, "r").read().splitlines()  

        return pos, neg, test

    if pp == 'normal':
        print('> loading preprocessed data in the format one tweet per line')
        PP_POS_PATH = './Data/preprocessed/pp_pos_otpl.txt'
        PP_NEG_PATH = './Data/preprocessed/pp_neg_otpl.txt'
        PP_TEST_PATH = './Data/preprocessed/pp_test_otpl.txt' 

        # load the data files = list with each line being a tweet
        pos = open(PP_POS_PATH, "r").read().splitlines()
        neg = open(PP_NEG_PATH, "r").read().splitlines()
        test = open(PP_TEST_PATH, "r").read().splitlines()

        return pos, neg, test

    