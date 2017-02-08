import pandas as pd
import numpy as np



hrc_train = pd.read_csv("./data/HRC_train.tsv", sep = "\t", names = ["id", "text"])
len(hrc_train)

#removes bad strings within a text
def removeBadStrings(oldstrings, rownumber):
    try:
        j = oldstrings.index("sent:")
        newstrings = oldstrings[j+5:]
        return removeBadStrings(newstrings, rownumber)
    except ValueError:
        return oldstrings

# print(hrc_train.loc[0, 'text'])
removeBadStrings(hrc_train.loc[0, 'text'], 0)

#identifies the index of a specific string within a text
def identify(text, string, index):
    try:
        j = text.index(str(string))
        newstrings = text[j + len(string):]
        return identify(newstrings, string, j)
    except ValueError:
        # print("the " + string + "was found in index number: " + str(index))
        return index

ind = np.zeros(len(hrc_train))
for i in range(len(hrc_train)):
    ind[i] = identify(hrc_train.loc[i, "text"], "subject:", 0)
print(len([i for i, j in enumerate(ind) if j == 0]))
# TODO: test hrc_train cleaning method
