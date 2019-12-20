#!/usr/bin/env python3
import re 
from typing import List

'''
python adaptation of the 'Ruby 2.0 script for preprocessing Twitter data' by Romain Paulus and modified by Jeffrey Pennington
https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
'''

def ruby_preprocessing(data):
	# Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # set patters that will be replaced in tweets 
    patterns = {
        r"<url>": "<url>",
        r"<user>": "<user>",
        r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes): "<smile>",
        r"{}{}p+".format(eyes, nose): "<lolface>",
        r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes): "<sadface>",
        r"{}{}[\/|l*]".format(eyes, nose): "<neutralface>",
        r"/": " / ",
        r"<3":"<heart>",
        r"[-+]?[.\d]*[\d]+[:,.\d]*": "<number>",
        r"#\S+": "<hashtag>",
        r"([!?.]){2,}": r"\1 <repeat>",
        r"\b(\S*?)(.)\2{2,}\b": r"\1\2 <elong>"
    }
    
    data_ = data
    for regex, replace in patterns.items():
        data_ = re.sub(regex, replace, data_)
        
    return data_


