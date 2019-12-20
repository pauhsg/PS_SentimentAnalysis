#!/usr/bin/env python3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from get_GloVe_emb_ML import *
from get_embeddings_ML import * 
from proj2_helpers import *

# set used scikit learn models and the used testsize 
MODELS = ['LR', 'SVM', 'NN']
TESTSIZE = 0.2

# useful paths
SUBMISSION_PATH = './Submissions/'

RUBY_FPOS_PATH = './Data/preprocessed/ruby_pos_otpl.txt' 
RUBY_FNEG_PATH = './Data/preprocessed/ruby_neg_otpl.txt' 
RUBY_TEST_PATH = './Data/preprocessed/ruby_test_otpl.txt'
RUBY_VECTORS_PATH = './Data/glove_pretrained/glove.twitter.27B.200d.txt'

GLOVE_VECTORS_PATH = './Data/glove'

COOC_PATH = './Data/produced/cooc.pkl'
VOC_PATH = './Data/produced/vocab.pkl'
EMBEDDINGS_PATH = './Data/produced/embeddings'
PP_POS_PATH = './Data/preprocessed/pp_pos_otpl.txt'
PP_NEG_PATH = './Data/preprocessed/pp_neg_otpl.txt'
PP_TEST_PATH = './Data/preprocessed/pp_test_otpl.txt'

#load pickle files
cooc_matrix = open_pickle_file(COOC_PATH)
vocabulary = open_pickle_file(VOC_PATH)

def split_standardize_pca(X, y, testsize):
    '''
    takes X matrix with mean words vectors and y the labels, splits them into train and test according 
    to the set testsize, performs StandardScaler() and PCA() and returns the resulting X_train, y_train, X_test,
    y_test and pca and scaler that were trained on X_train
    '''
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
    '''
    use scaler and pca that were fitted on train set and apply on test
    '''
    testx = scaler.transform(testx)
    testx = pca.transform(testx)
    return testx

def compute_accuracy(y_test, y_pred):
    '''
    compute accuracy
    '''
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    print('accuracy = ', accuracy)
    return accuracy

def prepare_glove_data_ML(pos, neg, test, vectors_filepath, dim_emb, testsize):
    '''
    gets all needed data with GloVe embeddings to perform ML
    '''
    vectors = get_vectors(vectors_filepath)
    full_df, X, y = get_train_emb(pos, neg, vectors, dim_emb)
    X_train, X_test, y_train, y_test, scaler, pca = split_standardize_pca(X, y, testsize)
    df_test, testx = get_test_emb(test, vectors, dim_emb)
    testx = std_pca_test(testx, scaler, pca)
    return X_train, X_test, y_train, y_test, testx, df_test

def prepare_we_data_ML(pos, neg, test, vocabulary, embeddings, dim_emb, testsize):
    '''
    gets all needed data with given embeddings to perform ML
    '''
    full_df, X, y = process_train_ML(pos, neg, vocabulary, embeddings, dim_emb)
    X_train, X_test, y_train, y_test, scaler, pca = split_standardize_pca(X, y, testsize)
    df_test, testx = process_test_ML(test, vocabulary, embeddings, dim_emb)
    testx = std_pca_test(testx, scaler, pca)
    return X_train, X_test, y_train, y_test, testx, df_test

def run_model(model, X_train, X_test, y_train, y_test, testx, df_test, submission_path):
    '''
    runs the selected model, saves the submission and returns accuracy 
    '''
    if model == 'LR':
        print('>>> RUNNING LOGISTIC REGRESSION')
        clf = linear_model.LogisticRegression(C=1e-2, max_iter=100000, n_jobs=-1).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = compute_accuracy(y_test, y_pred)
        y_pred_test = clf.predict(testx)
        create_submission(df_test, y_pred_test, submission_path)
        print('---> submission ready in Submissions folder')
        return acc

    if model == 'SVM':
        print('>>> RUNNING SUPPORT VECTOR MACHINES')
        svc = LinearSVC()
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        acc = compute_accuracy(y_test, y_pred)
        y_pred_test = svc.predict(testx)
        create_submission(df_test, y_pred_test, submission_path)
        print('---> submission ready in Submissions folder')
        return acc

    if model == 'NN':
        print('>>> RUNNING NEURAL NETWORK')
        clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(2, X_train.shape[1]), random_state=4, verbose=False, learning_rate='constant')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = compute_accuracy(y_test, y_pred)
        y_pred_test = clf.predict(testx)
        create_submission(df_test, y_pred_test, submission_path)
        print('---> submission ready in Submissions folder')
        return acc

def ruby_ML():
    '''
    for all embedding dimensions in DIM_EMB, runs all models of the list MODELS with embeddings 
    obtained with GloVe on data processed with Ruby 2.0 and returns accuracies in a 
    dictionnary f.ex: accuracies = {'LR': 0.73, 0.75, 0.77, 'SVM': ...}
    '''
    print('> testing glove embedding with Ruby preprocessing on different models with different embeddings dimensions')
    DIM_EMB = [50, 100, 200]
    
    pos = open(RUBY_FPOS_PATH, "r").read().splitlines()
    neg = open(RUBY_FNEG_PATH, "r").read().splitlines()
    test = open(RUBY_TEST_PATH, "r").read().splitlines()

    accuracies = dict()
    for dim in DIM_EMB:    
        print('---------------------------EMBEDDINGS DIMENSION: ', dim, '---------------------------')
        vec_path = RUBY_VECTORS_PATH + str(dim) + 'd.txt'
        X_train, X_test, y_train, y_test, testx, df_test = prepare_glove_data_ML(pos, neg, test, vec_path, dim, TESTSIZE)
        for model in MODELS:
            sub_path = SUBMISSION_PATH + model + '_ruby_' + str(dim) + 'd.CSV'
            acc = run_model(model, X_train, X_test, y_train, y_test, testx, df_test, sub_path)
            if model not in accuracies:
                accuracies[model] = acc
            elif type(accuracies[model]) == list:
                accuracies[model].append(acc)
            else: 
                accuracies[model] = [accuracies[model], acc]

    return accuracies

def glove_ML():
    '''
    for all embedding dimensions in DIM_EMB, runs all models of the list MODELS with embeddings 
    obtained with GloVe on data processed with our preprocessing and returns accuracies in a 
    dictionnary f.ex: accuracies = {'LR': 0.73, 0.75, 0.77, 'SVM': ...}
    '''
    print('> testing glove embedding with our preprocessing on different models with different embeddings dimensions')
    DIM_EMB = [50, 100, 200]

    pos = open(PP_POS_PATH, "r").read().splitlines()
    neg = open(PP_NEG_PATH, "r").read().splitlines()
    test = open(PP_TEST_PATH, "r").read().splitlines()

    accuracies = dict()
    for dim in DIM_EMB:    
        print('---------------------------EMBEDDINGS DIMENSION: ', dim, '---------------------------')
        vec_path = GLOVE_VECTORS_PATH + str(dim) + 'd/vectors.txt'
        X_train, X_test, y_train, y_test, testx, df_test = prepare_glove_data_ML(pos, neg, test, vec_path, dim, TESTSIZE)
        for model in MODELS:
            sub_path = SUBMISSION_PATH + model + '_glove_' + str(dim) + 'd.CSV'
            acc = run_model(model, X_train, X_test, y_train, y_test, testx, df_test, sub_path)
            if model not in accuracies:
                accuracies[model] = acc
            elif type(accuracies[model]) == list:
                accuracies[model].append(acc)
            else: 
                accuracies[model] = [accuracies[model], acc]

    return accuracies

def we_ML():
    '''
    for all embedding dimensions in DIM_EMB, runs all models of the list MODELS with embeddings 
    obtained with the given embedding solution (cooc.py + glove_solution) on data processed with 
    our preprocessing and returns accuracies in a dictionnary f.ex: accuracies = {'LR': 0.73, 0.75, 0.77, 'SVM': ...}
    '''
    print('> testing given embedding with our preprocessing on different models with different embeddings dimensions')
    DIM_EMB = [20, 50, 100]

    # load the data files = list with each line being a tweet
    pos = open(PP_POS_PATH, "r").read().splitlines()
    neg = open(PP_NEG_PATH, "r").read().splitlines()
    test = open(PP_TEST_PATH, "r").read().splitlines()

    accuracies = dict()
    for dim in DIM_EMB:
        print('---------------------------EMBEDDINGS DIMENSION: ', dim, '---------------------------')
        emb_path = EMBEDDINGS_PATH + str(dim) + 'd.npy'
        embeddings = np.load(emb_path)
        X_train, X_test, y_train, y_test, testx, df_test = prepare_we_data_ML(pos, neg, test, vocabulary, embeddings, dim, TESTSIZE)
        for model in MODELS:
            sub_path = SUBMISSION_PATH + model + '_we_' + str(dim) + 'd.CSV'
            acc = run_model(model, X_train, X_test, y_train, y_test, testx, df_test, sub_path)
            if model not in accuracies:
                accuracies[model] = acc
            elif type(accuracies[model]) == list:
                accuracies[model].append(acc)
            else: 
                accuracies[model] = [accuracies[model], acc]

    return accuracies














