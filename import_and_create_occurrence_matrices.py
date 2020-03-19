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

#   raw_file = open("recipes_raw_nosource_epi.json", 'r')
#   raw_file = re.sub("[^a-zA-Z ]+", "", raw_file.read())
#   global_dict = word_tokenize(raw_file)
#   print(global_dict)

with open("recipes_raw_nosource_epi.json") as file:
    data1 = json.load(file)

with open("recipes_raw_nosource_ar.json") as file:
    data2 = json.load(file)

with open("recipes_raw_nosource_fn.json") as file:
    data3 = json.load(file)


def merge(dict1, dict2):
    # Function that combines two dictionaries.
    res = {**dict1, **dict2}
    return res


data = data1

# Initialize the "tokenized" dictionary that will hold most of our text data.
data_tokenized = {}
# The NLTK works with "tokens." There are multiple token types, word or sentences for example.
# We will mostly use word tokens.


# Define a measurement keyword blacklist so that ingredient lists can be better analyzed.
measurement_blacklist = ['cup', 'cups', 'ml', 'tablespoon', 'tablespoons',
                         'teaspoon', 'teaspoons', 'tbsp', 'tsp', 'grams', 'g',
                         'ounce', 'ounces', 'oz', 'liter', 'l', 'jigger', 'pint',
                         'quart', 'gallon', 'half', 'quarter', 'pounds', 'to',
                         'large', 'small', 'medium', 'fine', 'garnish', 'lb',
                         'extralarge', 'inch', 'inches', 'chopped', 'cut', 'ADVERTISEMENT']

all_ingredient_names = []
ingredients = []
titles = []
title_lengths = []

i = 0
for key in data:  # iterate through every recipe!
    if 'ingredients' in data[key]:
        ing_temp = ''
        for ingredient in data[key]['ingredients']:
            ing_temp += ingredient + ' '

        title = data[key]['title']
        titles.append(title)
        ingredients.append(ing_temp)
        title_length = len(title)
        title_lengths.append(title_length)
        i += 1
    if i == 100:
        break

# these bad bois are going to create the basis vectors from all the
# words we throw at them
vectorizer_ing = CountVectorizer(min_df=0)
vectorizer_title = CountVectorizer(min_df=0)
# Define training & test sets.
titles_train = titles[:]
print(np.shape(titles))
titles_test = titles[:]
ing_train = ingredients[:]  # ingredients training set
ing_test = ingredients[:]  # ingredients test set

v_ing = vectorizer_ing.fit_transform(ing_train)
v_title = vectorizer_title.fit_transform(titles_train)
all_ingredient_words = vectorizer_ing.get_feature_names()
all_title_words = vectorizer_title.get_feature_names()

# print(vectorizer_ing.get_feature_names())
# print(v_ing.toarray().shape)

x_train = vectorizer_ing.fit_transform(ing_train).toarray()
x_test = vectorizer_ing.fit_transform(ing_test).toarray()
y_train = vectorizer_title.fit_transform(titles_train).toarray()
y_test = vectorizer_title.fit_transform(titles_test).toarray()

first = True
print(np.shape(x_train))
print(np.shape(y_train))
print("reshaping vectors\n")

i = 0
for row in np.arange(np.shape(x_train)[0]):
    time1 = time.process_time()
    print("progress: ", 100 * i / np.shape(x_train)[0], "%")
    # Define vectors relating to words, word frequencies, grammar, layout, etc...

    # x corresponds to the word megavector
    x = np.array(x_train[row].transpose())  # Transpose so we have a column to multiply.
    x = x.reshape(np.size(x), 1)  # Reshape for matrix multiplication
    y = np.array(y_train[row])
    y = y.reshape(1, np.size(y))
    y2 = np.array(title_lengths)
    y2 = y2.reshape(1, np.size(y2))

    if first:
        print("creating occurence matrix")
        occurrence_matrix = x * y  # Initial occurrences
        title_word_number_occurrence_matrix = x * y2
        first = False
    else:
        occurrence_matrix += x * y  # Sum up occurrences.
        title_word_number_occurrence_matrix += x * y2  # Sum up occurrences.
    time2 = time.process_time()
    print("iteration time: ", time2 - time1)
    i += 1

title_word_blacklist_indices = []
recipe_word_blacklist_indices = []

print("blacklisting words")
for word in all_ingredient_words:
    # loop through all words, and if they are on the blacklist, add their indices to the blacklist indices list
    w = word_tokenize(word)
    word_type = nltk.pos_tag(w)[0][1]
    print(word_type)
    if word_type != 'NN' or word in measurement_blacklist:
        #  Get indices of words that are not nouns or are measurements.
        recipe_word_blacklist_indices.append(all_ingredient_words.index(word))

for word in all_title_words:
    # loop through all words, and if they are on the blacklist, add their indices to the blacklist indices list
    w = word_tokenize(word)  # this may not be necessary, but NTLK likes its tokens
    word_type = nltk.pos_tag(w)[0][1]  # assign the part of speech of the given word. The [0][1]
    # just extracts the actual part of speech assignment
    print(word_type)
    if word_type != 'NN' or word in measurement_blacklist:
        #  Get indices of words that are not nouns or are measurements.
        title_word_blacklist_indices.append(all_title_words.index(word))  # append those to the blacklist list

occurrence_matrix_blacklisted = copy.deepcopy(occurrence_matrix)  # define a matrix that will be corrected by blacklist

for index in recipe_word_blacklist_indices:
    occurrence_matrix_blacklisted[index, :] = 0  # zero out any row that is blacklisted

for index in title_word_blacklist_indices:
    occurrence_matrix_blacklisted[:, index] = 0  # zero out any row that is blacklisted

# save occurrence matrices and keys for the matrix rows and columns
with open("title_word_key.pickle", "wb") as title_word_file:
    pickle.dump(all_title_words, title_word_file)
with open("ingredient_word_key.pickle", "wb") as ingredient_word_file:
    pickle.dump(all_ingredient_words, ingredient_word_file)
with open("occurrence_matrix.pickle", "wb") as om_file:
    pickle.dump(occurrence_matrix, om_file)
with open("occurrence_matrix_blacklisted.pickle", "wb") as omb_file:
    pickle.dump(occurrence_matrix_blacklisted, omb_file)




# create a copy so things can be deleted and not lose data from the main list
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


# NEED TO CREATE CORRELATION BETWEEN INGREDIENT WORDS AND TITLE LENGTH (NUMBER OF WORDS)


def distance_score(string1, string2):
    word_difference = list(set(string1) - set(string2))
    score = len(word_difference)
    return score
