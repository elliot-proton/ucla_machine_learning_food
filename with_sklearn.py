import json  # for dealing with json files
import re    # regular expression library
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
ing_test = ingredients[40:]   # ingredients test set

vectorizer_ing.fit(ing_train)
vectorizer_title.fit(titles_train)
print(vectorizer_ing.vocabulary_)
print(vectorizer_title.vocabulary_)
x_train = vectorizer_ing.transform(ing_train)
x_test = vectorizer_ing.transform(ing_test)
y_train = vectorizer_title.transform(titles_train)
y_test = vectorizer_title.transform(titles_test)

print(np.shape(x_train))
print(np.shape(y_train))

for i in np.arange(np.shape(x_train)[0]):
    for j in np.arange(np.shape(x_train)[1]):


# classifier = LogisticRegression()
# classifier.fit(x_train, y_train)
# classifier.fit(x_train, titles_train)

