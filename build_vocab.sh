#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ./Results/pp_pos_otpl_nd.txt ./Results/pp_neg_otpl_nd.txt | gsed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ./Results/vocab.txt
