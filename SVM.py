#!/usr/bin/env python3
import random
from datetime import datetime
from sklearn.metrics import f1_score

import numpy as np
import matplotlib.pyplot as plt

def hinge_loss(y, X, w):
    #print('hingle_loss')
    return np.clip(1 - y * (X @ w), 0, np.inf)

def calculate_primal_objective(y, X, w, lambda_):
    """compute the full cost (the primal objective), that is loss plus regularizer.
    X: the full dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_features)
    """
    v = hinge_loss(y, X, w)
    #print('primal obj')
    return np.sum(v) + lambda_ / 2 * np.sum(w ** 2)

def accuracy(y1, y2):
    #print('acc')
    return np.mean(y1 == y2)

def F1score(y1, y2):
    #print('f1')
    return f1_score(y2, y1)

def prediction(X, w):
    #print('pred')
    y_pred = X @ w
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred

def calculate_accuracy(y, X, w):
    """compute the training accuracy on the training set (can be called for test set as well).
    X: the full dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_features)
    """
    predicted_y = prediction(X, w)
    #print('compute acc')
    return accuracy(predicted_y, y)

def compute_F1score(y, X, w):
    predicted_y = prediction(X, w)
    #print('compute f1')
    return F1score(predicted_y, y)

def calculate_stochastic_gradient(y, X, w, lambda_, n, num_examples):
    """compute the stochastic gradient of loss plus regularizer.
    X: the dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_features)
    n: the index of the (one) datapoint we have sampled
    num_examples: N
    """
    # Be careful about the constant N (size) term!
    # The complete objective for SVM is a sum, not an average as in earlier SGD examples!
    def is_support(y_n, x_n, w):
        """a datapoint is support if max{} is not 0. """
        return y_n * x_n @ w < 1
    
    x_n, y_n = X[n], y[n]
    grad = - y_n * x_n.T if is_support(y_n, x_n, w) else np.zeros_like(x_n.T)
    grad = num_examples * np.squeeze(grad) + lambda_ * w
    return grad

def sgd_for_svm_demo(y, X):
    
    max_iter = 100000
    gamma = 1
    lambda_ = 0.01
    
    num_examples, num_features = X.shape
    w = np.zeros(num_features)
    
    for it in range(max_iter):
        # n = sample one data point uniformly at random data from x
        n = random.randint(0, num_examples-1)
        
        grad = calculate_stochastic_gradient(y, X, w, lambda_, n, num_examples)
        w -= gamma/(it+1) * grad
        
        if it % 10000 == 0:
            cost = calculate_primal_objective(y, X, w, lambda_)
            print("iteration={i}, cost={c}".format(i=it, c=cost))

    print("training accuracy = {l} \n training F1 score = {f}".format(l=calculate_accuracy(y, X, w), f=compute_F1score(y, X, w)))
    return w

def calculate_coordinate_update(y, X, lambda_, alpha, w, n):
    """compute a coordinate update (closed form) for coordinate n.
    X: the dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_examples)
    n: the coordinate to be updated
    """        
    # calculate the update of coordinate at index=n.
    x_n, y_n = X[n], y[n]
    old_alpha_n = np.copy(alpha[n])
    
    g = (y_n * x_n.dot(w) - 1)

    if old_alpha_n == 0:
        g = min(g, 0)
    elif old_alpha_n == 1.0:
        g = max(g, 0)
    else:
        g = g
    if g != 0:
        alpha[n] = min(
            max(old_alpha_n - lambda_ * g / (x_n.T.dot(x_n)), 0.0),
            1.0)
    
        # compute the corresponding update on the primal vector w
        w += 1.0 / lambda_ * (alpha[n] - old_alpha_n) * y_n * x_n
    return w, alpha


def calculate_dual_objective(y, X, w, alpha, lambda_):
    """calculate the objective for the dual problem."""
    return np.sum(alpha)  - lambda_ / 2.0 * np.sum(w ** 2) # w = 1/lambda * X * Y * alpha

def coordinate_descent_for_svm_demo(y, X):
    max_iter = 100000
    lambda_ = 0.01

    num_examples, num_features = X.shape
    w = np.zeros(num_features)
    alpha = np.zeros(num_examples)
    
    for it in range(max_iter):
        # n = sample one data point uniformly at random data from x
        n = random.randint(0,num_examples-1)
        
        w, alpha = calculate_coordinate_update(y, X, lambda_, alpha, w, n)
            
        if it % 10000 == 0:
            # primal objective
            primal_value = calculate_primal_objective(y, X, w, lambda_)
            # dual objective
            dual_value = calculate_dual_objective(y, X, w, alpha, lambda_)
            # primal dual gap
            duality_gap = primal_value - dual_value
            print('iteration=%i, primal:%.5f, dual:%.5f, gap:%.5f'%(
                    it, primal_value, dual_value, duality_gap))
            print("training accuracy = {l} \n training F1 score = {f}".format(l=calculate_accuracy(y, X, w), f=compute_F1score(y, X, w)))

