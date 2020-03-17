import json  # for dealing with json files
import re  # regular expression library
import pandas as pd
import nltk.corpus  # sample text for performing tokenization
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize  # Passing the string text into word tokenize for breaking the sentences

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
i = 0
for key in data:  # iterate through every recipe!
    # Lots of data cleaning/formatting.
    # Tokenize and convert to lower case with `.lower()`.
    title = word_tokenize(data[key]['title'].lower())
    # Convert token tuple list into just a list.
    title_pos = list(dict(nltk.pos_tag(title)).values())

    ingredients = data[key]['ingredients']
    ingredient_names = []
    for k in range(len(ingredients)):
        # The `re.sub` here removes everything except letters and spaces.
        ingredient_name = re.sub("[^a-zA-Z ]+", "", ingredients[k])
        # Strip off 'or' options
        ingredient_name = ingredient_name.split(" or ")[0]

        # Iterate through all ingredients in list and remove numbers and measurements.
        # Strip out any measurement references.
        # The `strip() removes leading and trailing whitespace - so easy!
        for l in measurement_blacklist:
            ingredient_name = (' ' + ingredient_name).lower().replace(' ' + l + ' ', '')

        tagged_name = dict((nltk.pos_tag(nltk.word_tokenize(ingredient_name))))
        ingredient_name = ''
        to_delete = []
        for m in tagged_name:
            if tagged_name[m] != 'NN':
                to_delete.append(m)

        # If it is not a noun, delete that entry
        for zebra in to_delete:
            del (tagged_name[zebra])

        for m in tagged_name:
            ingredient_name += ' ' + m
        ingredient_name = ingredient_name.strip()

        ingredient_names.append(ingredient_name)
        all_ingredient_names.append(ingredient_name)

    data_tokenized[key] = {}
    data_tokenized[key]['title'] = title
    data_tokenized[key]['title_pos'] = title_pos
    data_tokenized[key]['ingredient_names'] = ingredient_names
    i += 1
    if i == 10:
        break
    print(i)

# Get rid of duplicate names in `all_ingredient_names`.
all_ingredient_names = list(set(all_ingredient_names))
# Now we want to get the probability that a title word appears for a given ingredient word.
ingredient_title_counts = {}

for i_name in all_ingredient_names:
    ingredient_title_counts.update({i_name:{}})

#print(ingredient_title_counts)

for i_name in all_ingredient_names:
    for key in data_tokenized:
        for title_word in data_tokenized[key]['title']:
            if title_word in ingredient_title_counts[i_name]:
                ingredient_title_counts[i_name][title_word] += 1
            else:
                ingredient_title_counts[i_name].update({title_word:1})

print(ingredient_title_counts['butter'])
# print(data_tokenized)
