import copy
import time
import json  # for dealing with json files
import re  # regular expression library
import pandas as pd
import nltk.corpus  # sample text for performing tokenization
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize  # Passing the string text into word tokenize for breaking the sentenceg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import describe
import numpy as np
import pickle

with open("title_word_key.pickle", "rb") as title_word_file:
    # this key corresponds to each column title in the occurrence matrix
    title_word_key = pickle.load(title_word_file)
all_title_words = title_word_key
with open("ingredient_word_key.pickle", "rb") as title_word_file:
    # this key corresponds to each row title in the occurrence matrix
    ingredient_word_key = pickle.load(title_word_file)
all_ingredient_words = ingredient_word_key
with open("occurrence_matrix.pickle", "rb") as om_file:
    # this key corresponds to each row title in the occurrence matrix
    occurrence_matrix = pickle.load(om_file)
with open("occurrence_matrix_blacklisted.pickle", "rb") as omb_file:
    # this key corresponds to each row title in the occurrence matrix
    occurrence_matrix_blacklisted = pickle.load(omb_file)




omb = copy.deepcopy(occurrence_matrix_blacklisted)

'''store the indices of the matrix corrsponding to the ingredient word (row) and title word (col) that
occur the most frequently together'''
max_indices_ing_title = []

# same as above but ingredient word and number of words in the title (the length)
max_indices_ing_titlelength = []

number_of_maxes = 10  # the number of maxes to find and store in `max_indices`
for i in np.arange(10):  # this loop finds max index values in the occurrence matrix
    max_index = np.unravel_index(np.argmax(omb, axis=None), omb.shape)
    max_indices_ing_title.append(max_index)
    print(omb[max_index])
    omb[max_index] = 0  # set this value to zero now so that it doesn't get detected as a maximum

print(max_indices_ing_title)
print('testing')
# print(om[max_index])
# print(max_index)
for index in max_indices_ing_title:
    ing = all_ingredient_words[index[0]]
    title = all_title_words[index[1]]
    print(ing, '&', title, r'\\')


