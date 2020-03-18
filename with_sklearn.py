import copy
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

#   raw_file = open("recipes_raw_nosource_epi.json", 'r')
#   raw_file = re.sub("[^a-zA-Z ]+", "", raw_file.read())
#   global_dict = word_tokenize(raw_file)
#   print(global_dict)

with open("recipes_raw_nosource_epi.json") as file:
    data = json.load(file)

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
                         'extralarge', 'inch', 'inches']

all_ingredient_names = []
ingredients = []
titles = []
i = 0
for key in data:  # iterate through every recipe!
    ing_temp = ''
    for ingredient in data[key]['ingredients']:
        ing_temp += ingredient + ' '
    titles.append(data[key]['title'])
    ingredients.append(ing_temp)
    i += 1
    if i == 50:
        break

vectorizer_ing = CountVectorizer(min_df=0)
vectorizer_title = CountVectorizer(min_df=0)
# Define training & test sets.
titles_train = titles[:39]
titles_test = titles[40:]
ing_train = ingredients[:39]  # ingredients training set
ing_test = ingredients[40:]  # ingredients test set

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
for row in np.arange(np.shape(x_train)[0]):
    x = np.array(x_train[row].transpose())  # Transpose so we have a column to multiply.
    x = x.reshape(np.size(x), 1)            # Reshape for matrix multiplication
    y = np.array(y_train[row])
    y = y.reshape(1, np.size(y))

    if first:
        occurrence_matrix = x * y           # Initial occurrences
        first = False
    else:
        occurrence_matrix += x * y          # Sum up occurrences.

#max_indices = np.unravel_index(np.argpartition(occurrence_matrix, -5, axis=None)[-5:], occurrence_matrix.shape)
#print(max_indices[1][1])
max_index = np.unravel_index(np.argmax(occurrence_matrix, axis=None), occurrence_matrix.shape)

om = copy.deepcopy(occurrence_matrix)
max_indices = []
for i in np.arange(10): # this loop finds max index values in the occurrence matrix
    temp_max_index = np.unravel_index(np.argmax(om, axis=None), om.shape)
    max_indices.append(temp_max_index)
    print(om[temp_max_index])
    om[temp_max_index] = 0  # set this value to zero now so that it doesn't get detected as a maximum

print(max_indices)
print('testing')
#print(om[max_index])
#print(max_index)
for index in max_indices:
    print(index)
    ing = all_ingredient_words[index[0]]
    title = all_title_words[index[1]]
    print(ing, title)
