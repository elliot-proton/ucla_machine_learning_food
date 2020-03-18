import pandas as pd
import numpy as np
import nltk
import os
import json
import matplotlib.pyplot as plt
import nltk.corpus  # sample text for performing tokenization
from nltk.tokenize import word_tokenize  # Passing the string text into word tokenize for breaking the sentences
from nltk.probability import FreqDist


with open("recipes_raw_nosource_ar.json") as file:
       data = json.load(file)

titles = ''
i = 0
for key in data:
       try:
              title = (data[key]['title'])
              titles += ' ' + title
       except KeyError:
              print('KeyError, probably end of list')
       i += 1
       print(i)


print(nltk.pos_tag(title))
token = word_tokenize(titles)
fdist = FreqDist(token)
fdist1 = dict(fdist.most_common(10))
print(fdist1)

plt.bar(range(len(fdist1)), list(fdist1.values()), align='center')
plt.xticks(range(len(fdist1)), list(fdist1.keys()))
plt.show()

