#!/usr/bin/env python3
import nltk
import numpy as np
import re
import spacy
import wordninja
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from spellchecker import SpellChecker
from typing import List

lemmatizer = WordNetLemmatizer() 
spell = SpellChecker()
pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*') 
regex = re.compile('[^A-Za-zÀ-ÿ]')
sp = spacy.load('en_core_web_sm') 

smileys = {
    ':-)': '',
    ':)': '',
    ':-3': '',
    ':3': '',
    ':-]': '',
    ':]': '',
    ':->': '',
    ':>': '',
    ':-}': '',
    ':}': '',
    '=)': '',
    '=]': '',
    ':P': '',
    ':p': '',
    ':-P': '',
    ':D': '',
    ':-D': '',
    'xD': '',
    'XD': '',
    '=D': '',
    '=3': '',
    ':-(': '',
    ':(': '',
    ':-[': '',
    ':c': '',
    ':-c': '',
    ':[': '',
    ':<': '',
    ':-<': '',
    ':-{': '',
    ':{': '',
    ':@': '',
    ":'(": '',
    ":-'(": '',
    ":-')": '',
    ":')": '',
    ':-*': '',
    ':*': '',
    ':x': '',
    'xx': '',
    'xxx': '',
    'xo': '',
    'xoxo': '',
    'xoxoxo': '',
    'xoxoxox': '',
    '<3': '',
    ':-o': '',
    ':o': '',
    ':O': '',
    ':-O': '',
    ';)': '',
    ';-)': '',
}    
    
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
    return re.sub(r'(.)\1+', r'\1\1', s)

def expandContractions(text, c_re=c_re):
    
    def replace(match):
        return cList[match.group(0)]
    
    return c_re.sub(replace, text)

def parse_smileys(tweet: List[str]) -> List[str]:
    return [smileys.get(word, word) for word in tweet]

def preprocessing(data):
    
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

def remove_twt_duplicates(result):
    # remove tweets duplicates
    result_ = list(dict.fromkeys(result))
    return result_

def save_preprocessed_data(data, file_names, remove_dup=False):
    
    print('> saving datas')
    
    if (remove_dup==False):
        #save processed data in the format ['tweet1', 'tweet2', ...]
        with open(file_names[0], "w") as output:
            output.write(str(data))
        
        # save processed data in the format one tweet per line and no [] or ''
        File=open(file_names[1], 'w')
        for element in data:
            File.write(element)
            File.write('\n')
        File.close()
        
    else:
        
        print('> removing duplicates')
        
        # remove tweets duplicates
        data = list(dict.fromkeys(data))
        
        print('> saving datas w/out duplicates')
        
        #save processed data in the format ['tweet1', 'tweet2', ...]
        with open(file_names[2], "w") as output:
            output.write(str(data))
        
        # save processed data in the format one tweet per line and no [] or ''
        File_=open(file_names[3], 'w')
        for element in data:
            File_.write(element)
            File_.write('\n')
        File_.close()
    
    
    
    
    
    