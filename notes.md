## GOAL:
- Given a list of ingredients (or a full recipe), predict a good title for the recipe.
## IDEAS:
- Compute correlation coefficients between every word in a list of ingredients to each word in a title.
    - Create a dictionary with every word in the data set. The `value` for every word in the dictionary is the 
    corresponding title word distribution.
- Compute correlation coefficients between every word in a instruction set to each word in a title.
- Compute the correlation between the frequency of each word in a recipe and teach word in the title.
- Analyze verb correlation with sentence structure of titles.
- Weight ingredients based on amount.


- Somehow allow freedom to insert words into names that have not been seen before.
- Only use title grammars found in training data, to make results sensible.

- Utilize cross validation.
