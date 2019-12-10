#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat resultPtrain_nd.txt resultNtrain_nd.txt | gsed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab.txt
