#!/usr/bin/env python3
import nltk
import numpy as np
import re
import spacy
import string
import wordninja
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from spellchecker import SpellChecker
from typing import List

from ruby_python import *

lemmatizer = WordNetLemmatizer() 
spell = SpellChecker()
pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*') 
regex = re.compile('[^A-Za-zÀ-ÿ]')
sp = spacy.load('en_core_web_sm') 
table = str.maketrans(dict.fromkeys("[],'"))

# dictionnary of different smileys and the word with which we replace it in the text
smileys = {
    ':-)': 'happy',
    ':)': 'happy',
    ':-3': 'happy',
    ':3': 'happy',
    ':-]': 'happy',
    ':]': 'happy',
    ':->': 'happy',
    ':>': 'happy',
    ':-}': 'happy',
    ':}': 'happy',
    '=)': 'happy',
    '=]': 'happy',
    ':P': 'happy',
    ':p': 'happy',
    ':-P': 'happy',
    ':D': 'happy',
    ':-D': 'happy',
    'xD': 'happy',
    'XD': 'happy',
    '=D': 'happy',
    '=3': 'happy',
    ':-(': 'sad',
    ':(': 'sad',
    ':-[': 'sad',
    ':c': 'sad',
    ':-c': 'sad',
    ':[': 'sad',
    ':<': 'sad',
    ':-<': 'sad',
    ':-{': 'sad',
    ':{': 'sad',
    ':@': 'sad',
    ":'(": 'sad',
    ":-'(": 'sad',
    ":-')": 'happy',
    ":')": 'happy',
    ':-*': 'happy',
    ':*': 'happy',
    ':x': 'sad',
    'xx': 'happy',
    'xxx': 'happy',
    'xo': 'happy',
    'xoxo': 'happy',
    'xoxoxo': 'happy',
    'xoxoxox': 'happy',
    '<3': 'happy',
    ':-o': '',
    ':o': '',
    ':O': '',
    ':-O': '',
    ';)': 'happy',
    ';-)': 'happy',
}

# dictionnary of contractions like "don't" that will be replaced by the the entire expression -> "do not"
'''
List adapted to our case from:
https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
'''
cList = {
    "ain't": "am not",
      "aren't": "are not",
      "can't": "can not",
      "can't've": "can not have",
      "'cause": "because",
      "could've": "could have",
      "couldn't": "could not",
      "couldn't've": "could not have",
      "didn't": "did not",
      "doesn't": "does not",
      "don't": "do not",
      "hadn't": "had not",
      "hadn't've": "had not have",
      "hasn't": "has not",
      "haven't": "have not",
      "he'd": "he would",
      "he'd've": "he would have",
      "he'll": "he will",
      "he'll've": "he will have",
      "he's": "he is",
      "how'd": "how did",
      "how'd'y": "how do you",
      "how'll": "how will",
      "how's": "how is",
      "i'd": "I would",
      "i'd've": "I would have",
      "i'll": "I will",
      "i'll've": "I will have",
      "i'm": "I am",
      "i've": "I have",
      "isn't": "is not",
      "it'd": "it had",
      "it'd've": "it would have",
      "it'll": "it will",
      "it'll've": "it will have",
      "it's": "it is",
      "let's": "let us",
      "ma'am": "madam",
      "mayn't": "may not",
      "might've": "might have",
      "mightn't": "might not",
      "mightn't've": "might not have",
      "must've": "must have",
      "mustn't": "must not",
      "mustn't've": "must not have",
      "needn't": "need not",
      "needn't've": "need not have",
      "o'clock": "of the clock",
      "oughtn't": "ought not",
      "oughtn't've": "ought not have",
      "shan't": "shall not",
      "sha'n't": "shall not",
      "shan't've": "shall not have",
      "she'd": "she would",
      "she'd've": "she would have",
      "she'll": "she will",
      "she'll've": "she will have",
      "she's": "she is",
      "should've": "should have",
      "shouldn't": "should not",
      "shouldn't've": "should not have",
      "so've": "so have",
      "so's": "so is",
      "that'd": "that would",
      "that'd've": "that would have",
      "that's": "that is",
      "there'd": "there had",
      "there'd've": "there would have",
      "there's": "there is",
      "they'd": "they would",
      "they'd've": "they would have",
      "they'll": "they will",
      "they'll've": "they will have",
      "they're": "they are",
      "they've": "they have",
      "to've": "to have",
      "wasn't": "was not",
      "we'd": "we had",
      "we'd've": "we would have",
      "we'll": "we will",
      "we'll've": "we will have",
      "we're": "we are",
      "we've": "we have",
      "weren't": "were not",
      "what'll": "what will",
      "what'll've": "what will have",
      "what're": "what are",
      "what's": "what is",
      "what've": "what have",
      "when's": "when is",
      "when've": "when have",
      "where'd": "where did",
      "where's": "where is",
      "where've": "where have",
      "who'll": "who will",
      "who'll've": "who will have",
      "who's": "who is",
      "who've": "who have",
      "why's": "why is",
      "why've": "why have",
      "will've": "will have",
      "won't": "will not",
      "won't've": "will not have",
      "would've": "would have",
      "wouldn't": "would not",
      "wouldn't've": "would not have",
      "y'all": "you all",
      "y'alls": "you alls",
      "y'all'd": "you all would",
      "y'all'd've": "you all would have",
      "y'all're": "you all are",
      "y'all've": "you all have",
      "you'd": "you had",
      "you'd've": "you would have",
      "you'll": "you you will",
      "you'll've": "you you will have",
      "you're": "you are",
      "you've": "you have"
}    

c_re = re.compile('(%s)' % '|'.join(cList.keys()))


def clean_tweet(twt):
    '''
    takes a tweet (twt) and performs several cleaning steps:
        - when there is a hashtag like #iloveML will transform it into "i love ML"
        - replaces smileys with a word using the above dictionnary 'smileys'
        - expand contractions like don't -> do not using the above dictionnary cList
        - correct elongated words like haaaaappppyyyy -> happy
    and returns the cleaned tweet
    '''
    # get words in hashtags and removes smiley
    tweet = [' '.join(wordninja.split(word)) if word.startswith("#") \
             else (smileys.get(word) if word in smileys else word) for word in twt.split()]
    tweet=' '.join(tweet)
    
    # expand contractions like don't -> do not    
    tweet = expandContractions(tweet, c_re=c_re)

    # correct the elongated words
    tweet = [remove_consecutive_dups(word) if len(spell.unknown([remove_consecutive_dups(word)])) == 0 \
            else (spell.correction(remove_consecutive_dups(word))) for word in tweet.split()]

    tweet = ' '.join(tweet)
    
    return tweet


def get_wordnet_pos(treebank_tag):
    '''
    POS-tags furnished by the nltk POS-tag function are not in the same format that the ones used by the lemmatizer
    --> this function converts them
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def remove_consecutive_dups(s):
    '''
    removes consecutive duplication of letters in elongated words like 'haaaappppyyy'
    '''
    return re.sub(r'(.)\1+', r'\1\1', s)

def expandContractions(text, c_re=c_re):
    '''
    expands contractions like don't -> do not
    '''
    def replace(match):
        return cList[match.group(0)]
    
    return c_re.sub(replace, text)

def parse_smileys(tweet: List[str]) -> List[str]:
    '''
    replaces smiley with words using the above list 'smileys'
    '''
    return [smileys.get(word, word) for word in tweet]

def remove_ids(test_data):
    '''
    removes ids in the test data 
    '''
    return [twt.split(',', 1)[-1] for twt in test_data]

def preprocessing(data):
    '''
    this function allows to perform all our preprocessing steps described in the functions above
    on given data 
    '''
    print('> running preprocessing pipeline')
    
    # remove empty tweets
    data = [twt for twt in data if not (twt.isspace() or twt == np.nan or not twt)]
    
    # remove tags
    data = [twt.replace('<user>', '') for twt in data]
    data = [twt.replace('<url>', '') for twt in data]
    
    # convert into lower cases
    data = [twt.lower() for twt in data] 
    
    # take from word from hashtag, removes smileys and corrects elongated words
    data = [clean_tweet(twt) for twt in data]
    
    # remove all non-alphabetical characters 
    data = [regex.sub(' ', twt) for twt in data]
    
    # remove empty tweets
    data = [twt for twt in data if not (twt.isspace() or twt == np.nan or not twt)]
    
    # substitute multiple whitespace with single whitespace 
    data = [' '.join(twt.split()) for twt in data]
    
    # remove stopwords
    data = [pattern.sub('', twt) for twt in data]
    
    # remove words that are < 2 letters
    data = [[token for token in twt.split() if len(token) > 2] for twt in data]
    data = [' '.join(twt) for twt in data]
    
    # tokenize
    data = [sp(twt) for twt in data]
    
    # get POS-tag of words of each tweet
    data = [[(token.text, token.tag_) for token in twt]  for twt in data]
    
    # transform POS tag + lemmatize
    data = [[lemmatizer.lemmatize(twt[i][0], get_wordnet_pos(twt[i][1])) for i in range(len(twt))] for twt in data]
    
    # back to tweet format
    data = [' '.join(twt) for twt in data]
    
    # remove words that are < 2 letters
    data = [[token for token in twt.split() if len(token) > 2] for twt in data]
    data = [' '.join(twt) for twt in data]
    
    return data 

def remove_twt_duplicates(data):
    '''
    removes tweets that are duplicated in the data
    '''
    return list(dict.fromkeys(data))

def save_as_list(data, file_path):
    '''
    save processed data in the format ['tweet1', 'tweet2', ...] -> as a list
    '''
    print("> saving datas in the format ['tweet1', 'tweet2', ...]")
    with open(file_path, "w") as output:
        output.write(str(data))
        
def save_as_otpl(data, file_path):
    '''
    save processed data in the format one tweet per line and no [] or '' -> one tweet per line = otpl
    '''
    print("> saving datas in the format one tweet per line and no [] or '' ")
    File = open(file_path, 'w')
    for element in data:
        File.write(element)
        File.write('\n')
    File.close()

def convert(lst): 
    '''
    converts a list of tweets like ['tweet1', 'tweet2', ...] into one single string 'tweet1 tweet2 ...'
    '''
    return str(lst).translate(table) 

def corpus_for_glove(pos_otpl, neg_otpl, file_path):
    '''
    GloVe needs the data to be in a certain format: pos and neg together in a .txt file,
    with all tweets converted into words seperated by whitespaces -> that is what this 
    function does + it saves the obtained corpus in a file
    '''
    print("> prepare corpus for GloVe with pp_pos and pp_neg")
    # create list with one token per line for pp_pos
    pp_pos_g = [twt.split() for twt in pos_otpl]
    pp_pos_g = [token for twt in pp_pos_g for token in twt]
    pos_corpus = convert(pp_pos_g)

    # create list with one token per line for pp_neg
    pp_neg_g = [twt.split() for twt in neg_otpl]
    pp_neg_g = [token for twt in pp_neg_g for token in twt]
    neg_corpus = convert(pp_neg_g)

    # concatenate both lists
    corpus_g = pos_corpus + ' ' + neg_corpus
    print('corpus length:', len(corpus_g))

    print("> saving corpus")
    text_file = open(file_path, "w")
    text_file.write(corpus_g)
    text_file.close()

def run_train_preprocessing(pos, neg, pp):
    '''
    runs specified preprocessing ('pp' argument) on pos and neg train sets and saves them in .txt files with and without tweet duplicates
    '''
    if pp == 'normal':
        print('> running normal preprocessing')
        # run preprocessing on train_pos.txt and save
        pp_pos = preprocessing(pos)
        save_as_list(pp_pos, './Data/preprocessed/pp_pos_list.txt')
        save_as_otpl(pp_pos, './Data/preprocessed/pp_pos_otpl.txt')

        # remove duplicates and save
        pp_pos_ = remove_twt_duplicates(pp_pos)
        save_as_list(pp_pos_, './Data/preprocessed/pp_pos_list_nd.txt')
        save_as_otpl(pp_pos_, './Data/preprocessed/pp_pos_otpl_nd.txt')

        # run preprocessing on train_neg.txt and save 
        pp_neg = preprocessing(neg)
        save_as_list(pp_neg, './Data/preprocessed/pp_neg_list.txt')
        save_as_otpl(pp_neg, './Data/preprocessed/pp_neg_otpl.txt')

        # remove duplicates and save
        pp_neg_ = remove_twt_duplicates(pp_neg)
        save_as_list(pp_neg_, './Data/preprocessed/pp_neg_list_nd.txt')
        save_as_otpl(pp_neg_, './Data/preprocessed/pp_neg_otpl_nd.txt')
    
    if pp == 'ruby':
        print('> running Ruby 2.0 preprocessing')
        # run preprocessing on train_pos.txt and save
        pp_pos = [ruby_preprocessing(line) for line in pos]
        save_as_list(pp_pos, './Data/preprocessed/ruby_pos_list.txt')
        save_as_otpl(pp_pos, './Data/preprocessed/ruby_pos_otpl.txt')

        # remove duplicates and save
        pp_pos_ = remove_twt_duplicates(pp_pos)
        save_as_list(pp_pos_, './Data/preprocessed/ruby_pos_list_nd.txt')
        save_as_otpl(pp_pos_, './Data/preprocessed/ruby_pos_otpl_nd.txt')

        # run preprocessing on train_neg.txt and save 
        pp_neg = [ruby_preprocessing(line) for line in neg]
        save_as_list(pp_neg, './Data/preprocessed/ruby_neg_list.txt')
        save_as_otpl(pp_neg, './Data/preprocessed/ruby_neg_otpl.txt')

        # remove duplicates and save
        pp_neg_ = remove_twt_duplicates(pp_neg)
        save_as_list(pp_neg_, './Data/preprocessed/ruby_neg_list_nd.txt')
        save_as_otpl(pp_neg_, './Data/preprocessed/ruby_neg_otpl_nd.txt')

    
def run_test_preprocessing(test, pp):
    '''
    runs specified preprocessing ('pp' argument) on test set and saves it in .txt files 
    '''
    if pp == 'normal':
        print('> running normal preprocessing')
        test = remove_ids(test)
        pp_test = preprocessing(test)
        save_as_list(pp_test, './Data/preprocessed/pp_test_list.txt')
        save_as_otpl(pp_test, './Data/preprocessed/pp_test_otpl.txt')

    if pp == 'ruby':
        print('> running Ruby 2.0 preprocessing')
        test = remove_ids(test)
        pp_test = [ruby_preprocessing(line) for line in test]
        save_as_list(pp_test, './Data/preprocessed/ruby_test_list.txt')
        save_as_otpl(pp_test, './Data/preprocessed/ruby_test_otpl.txt')
    



    


        

    