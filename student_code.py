from __future__ import division

#############################################################################
# Copyright 2011 Jason Baldridge
# 
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#############################################################################

# Imports from Python standard libraries
import re,math
from operator import itemgetter

# Imports from external packages included in the same directory
from porter_stemmer import PorterStemmer
import twokenize
# import emoticons

# Imports from other packages created for this homework
from classify_util import makefeat, window

#############################################################################
# Code to set up some resources for your features

# Create the stemmer for later use.
stemmer = PorterStemmer()

# Read in the stop words
stop_words = set([x.strip() for x in open("data/resources/stopwords.english", encoding="ISO-8859-1").readlines()])

# Obtain the words from Opinion Lexicon
neg_words =  set([x.strip() for x in open("data/resources/negative-words.txt", encoding="ISO-8859-1").readlines() if not(x.startswith(";"))])
neg_words.remove("")

pos_words =  set([x.strip() for x in open("data/resources/positive-words.txt", encoding="ISO-8859-1").readlines() if not(x.startswith(";"))])
pos_words.remove("")


#############################################################################
# Add your regular expressions here.


#############################################################################
# General features (for both subjectivity and polarity)
def extract_features (post, extended_features):
    features = []

    # FIXME: Change to use tokenization from twokenize.
    tokens = twokenize.tokenize(post.content)

    # FIXME: lower case the tokens
    lctokens = [token.lower() for token in tokens]

    # FIXME: Create stems here
    stems = [stemmer.stem_token(token) for token in lctokens]

    # Add unigram features. 
    # FIXME: exclude stop words
    filtered_tokens = [token for token in stems if token not in stop_words]

    # FIXME: consider using lower case version and/or stems
    ## filtered_tokens is the result of lowercase-ing, stemming, and excluding stop words
    features.extend([makefeat("word",tok) for tok in filtered_tokens])

    # The same thing, using a for-loop (boooring!)
    #for tok in tokens:
    #    features.append(makefeat("word",tok))
        
    if extended_features:
        # FIXME: Add bigram features (suggestion: use the window function in classify_utils.py

        ## window() outputs a tuple and makefeat() takes in a single string, so we need to convert the output to a string

        ## this causes an error I wasn't able to fix, so I've commented it out

        #bigrams = list(window(filtered_tokens))

        #bigram_list = []
        #for bigram in bigrams:
            #tmp_string = ""
            #for word in bigram:
                #tmp_string = tmp_string + word
            #bigram_list.append(tmp_string)

        #features.append([makefeat("bigram", bigram) for bigram in bigram_list])
    
        ## Polarity Lexicon
        ## Copied code from lexicon ratio baseline function
        num_positive = 0
        num_negative = 0
        for token in filtered_tokens:
            if token in neg_words:
                num_negative += 1
            elif token in pos_words:
                num_positive += 1
        
        num_neutral = .2
        if num_positive == num_negative:
            num_neutral += len(tokens)
        num_positive += .1
        num_negative += .1
        denominator = num_positive + num_negative + num_neutral

        predictions = [("positive", num_positive/denominator), 
                       ("negative", num_negative/denominator),
                       ("neutral", num_neutral/denominator)]
        
        predictions.sort(key=itemgetter(1),reverse=True)

        sentiment = predictions[0][0]

        features.extend([makefeat("sentiment", sentiment)])
    
        ## Presence of curse words

        ## Initialize list of curse words
        curse_words = ["fuck", "shit", "bitch", "crap", "damn", "prick", "asshole", "dick", "bastard"]

        ## Initialize curse words count to 0
        num_curse = 0

        ## Add 1 for each instance of curse words in post
        for token in tokens:
            if token in curse_words:
                num_curse += 1
        
        ## Assign label based on count
        if num_curse > 0:
            curse = "vulgar"
        else:
            curse = "not-vulgar"
        
        features.extend([makefeat("vulgarity", curse)])

        ## Other available variables
        features.extend([makefeat("op_gender", post.op_gender)])
        features.extend([makefeat("target_gender", post.target_gender)])
        features.extend([makefeat("experience", post.experience)])
        features.extend([makefeat("relationship", post.relationship)])
        features.extend([makefeat("industry", post.industry)])
        features.extend([makefeat("condition", post.condition)])
        features.extend([makefeat("action", post.action)])
        features.extend([makefeat("intention", post.intention)])
        features.extend([makefeat("impact", post.impact)])
        features.extend([makefeat("good_standing", post.good_standing)])
        features.extend([makefeat("perspective", post.perspective)])

    return features


#############################################################################
# Predict sentiment based on ratio of positive and negative terms in a post
def majority_class_baseline (postset):

    # FIXME: Compute the most frequent label in postset and return it

    ## Initialize empty dictionary to hold label counts
    label_counts = {}

    ## Loop over each post in postset to retrieve label and add to 
    ## label_counts
    for post in postset:
        label = post.label
        label_counts[label] = label_counts.get(label, 0) + 1

    ## Set this to equal the most common label
    majority_class_label = max(label_counts, key=label_counts.get)

    #print(label_counts)

    return majority_class_label


#############################################################################
# Predict sentiment based on ratio of positive and negative terms in a post
def lexicon_ratio_baseline (post):

    # FIXME: Change to use tokenization from twokenize
    tokens = twokenize.tokenize(post.content)

    # FIXME: Count the number of positive and negative words in the post
    
    ## Initialize num_positive and num_negative to 0 to start
    num_positive = 0
    num_negative = 0    

    ## Loop over each token and check to see if it exists in neg_words or
    ## pos_words, add 1 to respective count
    for token in tokens:
        if token in neg_words:
            num_negative += 1
        elif token in pos_words:
            num_positive += 1


    #########################################################################
    # Don't change anything below this comment
    #########################################################################

    # Let neutral be prefered if nothing is found.
    num_neutral = .2

    # Go with neutral if pos and neg are the same
    if num_positive == num_negative:
        num_neutral += len(tokens)

    # Add a small count to each so we don't get divide-by-zero error
    num_positive += .1
    num_negative += .1

    denominator = num_positive + num_negative + num_neutral

    # Create pseudo-probabilities based on the counts
    predictions = [("positive", num_positive/denominator), 
                   ("negative", num_negative/denominator),
                   ("neutral", num_neutral/denominator)]

    # Sort
    predictions.sort(key=itemgetter(1),reverse=True)

    # Return the top label and its confidence
    return predictions[0]
    
