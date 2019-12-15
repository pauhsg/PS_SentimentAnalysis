#!/usr/bin/env python3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

def split_standardize_pca(X, y, testsize):

    print('> splitting datas into train and test sets')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=1)

    print('> standardizing')
    # create instance of StandardScaler
    scaler = StandardScaler()
    # fit on train set only
    scaler.fit(X_train)
    # apply transform to train and test
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print('> performing PCA')
    # create instance of PCA
    pca = PCA(.95)
    # fit PCA on train set only
    pca.fit(X_train)
    # apply on train and test 
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    print('Train set: X shape: ', X_train.shape, 'y shape:', y_train.shape)
    print('Test set: X shape: ', X_test.shape, 'y shape:', y_test.shape)

    return X_train, X_test, y_train, y_test, scaler, pca

def std_pca_test(testx, scaler, pca):
    # use scaler and pca that were fitted on train set and apply on test
    testx = scaler.transform(testx)
    testx = pca.transform(testx)
    return testx

def compute_accuracy(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    print(accuracy)
