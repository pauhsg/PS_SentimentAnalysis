#!/usr/bin/env python3

from ML_sklearn import *

'''
This files allows to run LR, SVM and NN models on:
    - 'normal' preprocessing on train_pos.txt and train_neg.txt, with given word embeddings in dimensions 20, 50 and 100
    - 'normal' preprocessing on train_pos.txt and train_neg.txt, with GloVe word embeddings in dimensions 50, 100, 200
    - 'ruby' preprocessing on full_train_pos.txt and full_train_neg.txt with GloVe pre-trained downloaded word embeddings in dimensions 50, 100, 200
and stores the corresponding accuracies in a dict. 
'''

accuracies_we_ML = we_ML()
accuracies_glove_ML = glove_ML()
accuracies_ruby_ML = ruby_ML()




