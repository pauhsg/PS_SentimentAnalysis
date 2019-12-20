# Tweet Sentiment Analysis

Project 2 of the Machine Learning course at EPFL sets the challenge to generate a model that could predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.

## Dataset

1. Using your epfl.ch email go to https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019 to register for the challenge and download the data set in the tab `Resources`. 
2. Unzip the `twitter-datasets` folder and store it in the `Data` folder of the project repository.

## Installations

Several libraries are used for this project and need to be installed. In your terminal do the following command:

pip install -r requirements.txt

Some files also need to be downloaded:

- If you want to use pre-trained Twitter vectors and preprocess your data with Ruby 2.0:
    1. Go to https://nlp.stanford.edu/projects/glove/ and in the `Download pre-trained word vectors` section click on the link `glove.twitter.27B.zip`, rename the downloaded folder to `glove_pretrained` and put it into the folder `Data` of the project repository.
    (Ruby 2.0 script link is found just below on https://nlp.stanford.edu/projects/glove/, but the python-adapted version is available here as ruby_python.py)

- If you want to train your own GloVe vectors on your own preprocessed data: 
    1. Go to https://nlp.stanford.edu/projects/glove/ and in the `Getting started` section click on the link `Download the code (licensed under the Apache License, Version 2.0)` and put the downloaded `Glove-1.2 folder` into the folder `Data` of the project repository.
    2. Follow the next instructions to get your word embeddings, and before running ./demo.sh, open `demo.sh` and:
        - choose the embedding dimension you want to get (variable: VECTOR_SIZE)
        - put the correct path to the corpus you want to use (variable: CORPUS)
    3. When your embeddings are obtained, store them in a new folder called for instance `glove50d` if you chose VECTOR_SIZE=50 and put it into the folder `Data`.

- If you want to use Google Word2Vec open your browser and paste https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz. It will directly dowload the folder containing pre-trained word vectors. Put the downloaded `GoogleNews-vectors-negative300.bin` directly in a new folder `google_pretrained` and put it into de folder `Data`.

## Run the project 

To run the project, simply run the file `run.py`.

*If you want to run LR, SVM, NN and CNN with Word2Vec embeddings run `W2V_LR.py`, `W2V_SVM.py`, `W2V_NN.py`, `W2V_CNN.py` files.*
*If you want to run LR, SVM, NN on given embeddings and GloVe embeddings with 'normal' preprocessing, or on GloVe pre-trained word vectors with 'ruby' preprocessing, run `test_models.py`*

*Note: LR = logistic regression, SVM = support vector machines, NN = neural networks, CNN = Convolutional Neural Networks*

## Organisation of the repository

If you performed all the installation steps, your repository should have this organisation. 

```
| 
|   build_vocab.sh                                    > allows to build a vocabulary from .txt files specified in it
|   cooc.py                                           > produces the co-occurence matrix of files specified in it
|   cut_vocab.sh                                      > cuts the obtained vocab.txt
|   get_embeddings_ML.py                              > contains functions to get embeddings in the right format to perform ML
|   get_GloVe_emb_ML.py                               > contains functions to get GloVe embeddings in the right format to perform ML
|   glove_solution.py                                 > takes the sparse cooc matrix and transforms it in dense embeddings
|   load_data.py                                      > contains functions to load preprocessed data
|   ML_CNN_keras.py                                   > contains functions allowing to perform keras CNN              
|   ML_sklearn.py                                     > contains functions allowing to perform ML with sklearn
|   pickle_vocab.py                                   > transform vocab.txt into pickle file vocab.pkl 
|   proj2_helpers.py                                  > contains helpers functions to create submissions and open files
|   README.md                                         > README of the project
|   requirements.txt                                  > contains required libraries to be able to run the project
|   ruby_python.py                                    > contains python-adapted version of Ruby 2.0 tweets preprocessing (pp='ruby')
|   run.py                                            > file to run the model that yields the best result on AIcrowd
|   test_models.py                                    > allows to run LR, SVM and NN on different datas with different word emb. and dim.
|   twts_preprocess.py                                > functions to perform our preprocessing (pp='normal')
|   W2V_CNN.py                                        > run this file to perform CNN using Word2Vec embeddings
|   W2V_LR.py                                         > run this file to perform LR using Word2Vec embeddings
|   W2V_NN.py                                         > run this file to perform NN using Word2Vec embeddings
|   W2V_SVM.py                                        > run this file to perform SVM using Word2Vec embeddings
|   W2V_utilities.py                                  > contains helper functions for ML models using Word2Vec embeddings
|
+---Data
|
|   +---glove_pretrained     
|       glove.twitter.27B.25d.txt                     > downloaded GloVe pre-trained twitter word embeddings with dim_embeddings = 25
|       glove.twitter.27B.50d.txt                     > downloaded GloVe pre-trained twitter word embeddings with dim_embeddings = 50      
|       glove.twitter.27B.100d.txt                    > downloaded GloVe pre-trained twitter word embeddings with dim_embeddings = 100
|       glove.twitter.27B.200d.txt                    > downloaded GloVe pre-trained twitter word embeddings with dim_embeddings = 200
|
|   +---glove50d     
|       vectors.txt                                   > GloVe word embeddings with VECTOR_SIZE=50 in demo.sh
|       vocab.txt                                     > GloVe vocabulary   
|  
|   +---glove100d     
|       vectors.txt                                   > GloVe word embeddings with VECTOR_SIZE=100 in demo.sh
|       vocab.txt                                     > GloVe vocabulary   
|  
|   +---glove200d     
|       vectors.txt                                   > GloVe word embeddings with VECTOR_SIZE=200 in demo.sh
|       vocab.txt                                     > GloVe vocabulary  
|
|   +---google_pretrained
|       GoogleNews-vectors-negative300.bin            > downloaded Word2Vec Google pre-trained word embeddings  
|
|   +---preprocessed   
|       corpus_glove.txt.txt                          > corpus ready to be used for GloVe embeddings                                      
|       pp_neg_otpl_nd.txt                            > 'normal' preprocessing applied on train_neg.txt saved as one tweet per line
|       pp_pos_otpl_nd.txt                            > 'normal' preprocessing applied on train_pos.txt saved as one tweet per line
|       pp_test_otpl.txt                              > 'normal' preprocessing applied on test_data.txt saved as one tweet per line
|       ruby_neg_otpl.txt                             > 'ruby' preprocessing applied on train_neg.txt saved as one tweet per line
|       ruby_pos_otpl.txt                             > 'ruby' preprocessing applied on train_pos.txt saved as one tweet per line
|       ruby_test_otpl.txt                            > 'ruby' preprocessing applied on test_data.txt saved as one tweet per line
|                
|   +---produced                                         
|       cooc.pkl                                      > co-occurence matrix produced by running cooc.py 
|       embeddings20d.npy                             > embeddings produced by running glove_solution.py with dim_embeddings = 20  
|       embeddings50d.npy                             > embeddings produced by running glove_solution.py with dim_embeddings = 50 
|       embeddings100d.npy                            > embeddings produced by running glove_solution.py with dim_embeddings = 100 
|       vocab_cut.txt                                 > cut vocabulary produced by running cut_vocab.sh 
|       vocab.pkl                                     > pickle vocabulary produced by running pickle_vocab.py
|       vocab.txt                                     > vocabulary produced by running build_vocab.sh
|                
|   +---twitter-datasets
|       sample-submission.csv                         > example of sample submission file in the correct format
|       test_data.txt                                 > test set containing 10 000 unlabeled tweets
|       train_neg_full.txt                            > training set containing 1 250 0000 negative tweets   
|       train_neg.txt                                 > training set containing 100 0000 negative tweets
|       train_pos_full.txt                            > training set containing 1 250 0000 positive tweets  
|       train_pos.txt                                 > training set containing 100 000 positive tweets
|                  
+---Submissions
|
|   CNN_ruby_200d.csv                                 > best submission on AIcrowd using CNN on ruby preprocessed data with DIM_EMB = 200   
|                                          
```  

## References 

GloVe:
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation 
- Romain Paulus, Jeffrey Pennington. Ruby 2.0: Script for preprocessing tweets

Convolutional Neural Network and Word2Vec embeddings:
- Saad Arshad. 20 Sept 2019. Sentiment Analysis / Text Classification Using CNN (Convolutional Neural Network).
  https://towardsdatascience.com/cnn-sentiment-analysis-1d16b7c5a0e7
  https://github.com/saadarshad102/Sentiment-Analysis-CNN

## Note

The embeddings obtained with the given course material are already stored in the folder `./Data/produced`




