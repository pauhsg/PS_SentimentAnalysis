#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt


def main():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    data, row, col = [], [], []
    counter = 1
    for fn in ['./Datasets/twitter-datasets/pos_train.txt' , './Datasets/twitter-datasets/neg_train.txt' ]:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
                
    print("\n nb unique val in row: ",len(Counter(row).keys())) # equals to list(set(words))
    print("\n nb unique val in col",len(Counter(col).keys()))
    print("taille de row:",len(row))
    print("\n taille de col:",len(col))
    print("\n taille de data:",len(data)) 
    
    cooc = coo_matrix((data, (row, col)))
    
    print("FIRST CoOC de taille:",cooc.shape)
    print("summing duplicates (this can take a while)")
    plt.spy(cooc, markersize=0.01)
    
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    
    print("SENCOND CoOC de taille:",cooc.shape)
    
    with open('cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
    
    plt.spy(cooc, markersize=0.01)


if __name__ == '__main__':
    main()
