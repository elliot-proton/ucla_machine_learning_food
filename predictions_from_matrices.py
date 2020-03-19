import copy
import time
import os
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

os.chdir('/home/elliot/PycharmProjects/ucla_machine_learning_food/small_data')

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
'''store the indices of the matrix corresponding to the ingredient word (row) and title word (col) that
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
for index in max_indices_ing_title:
    ing = all_ingredient_words[index[0]]
    title = all_title_words[index[1]]
    print(ing, '&', title, r'\\')

"""
To get the words of highest probability, sum all the rows representing the given ingredient words. The columns with the 
highest N counts (where N is the number of words that you want in your title) represent the most likely title words. 
Retrieve those words and combine them to make a title.
"""

# Define a recipe to test.
test_list = ['egg', 'flour', 'chicken']


def produce_title(ingredient_list, title_length, occurrence_mat, ing_key, title_key):
    il = ingredient_list
    om = copy.deepcopy(occurrence_mat)
    ingredient_indices = []
    for ingredient in ingredient_list:
        ing_index = ing_key.index(ingredient)
        print(ing_index)
        ingredient_indices.append(ing_index)
    # Now sum up all the rows corresponding to the ingredients.
    title_likelihood = np.sum(occurrence_matrix_blacklisted[ingredient_indices], axis=0)
    # Create a copy to freely manipulate.
    tl = copy.deepcopy(title_likelihood)
    # Define a desired title length.
    title_word_indices = []
    # Need to normalize counts to get probabilities? TODO
    for i in np.arange(title_length):  # this loop finds max index values in the occurrence matrix
        max_index = np.argmax(tl)
        title_word_indices.append(max_index)
        tl[max_index] = 0  # set this value to zero now so that it doesn't get detected as a maximum

    title_words = []
    for index in title_word_indices:
        title_words.append(title_word_key[index])

    return title_words


print('fried' in ingredient_word_key)
print(produce_title(test_list, 3, occurrence_matrix_blacklisted, ingredient_word_key, title_word_key))
