#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat result_test.txt | gsed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_text.txt
