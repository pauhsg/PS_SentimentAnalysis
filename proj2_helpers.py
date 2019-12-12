#!/usr/bin/env python3
import subprocess 
from pickle_vocab import *
from cooc import *
from glove_solution import *


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
        
        
def open_pickle_file(file_path):    
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj
            
                
# def create_dense_embeddings(txt_file):
#     print('> running build_vocab.sh')
#     subprocess.call(['./build_vocab.sh'])
    
#     print('> running cut_vocab.sh')
#     subprocess.call(['./cut_vocab.sh'])
    
#     print('> running pickle_vocab')
#     pickle_vocab()
    
#     print('> running cooc')
#     cooc()
    
#     print('> running glove_solution')
#     glove_solution(embedding_dim = 20)
    
    
    
    
    