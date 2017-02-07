import pandas as pd
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import string as st
import re

hrc_train = pd.read_csv("./data/HRC_train.tsv", sep = "\t", names = ["id", "text"])


badstrings = ["unclassified u.s. department of state", "case no. ............",
                "doc no. c........", "date: ..........","state dept. . produced to house select",
                "subject to agreement on sensitive information & redactions.","no foia waiver state...........",
                "no foia waiver.",  "unclassified us department of state"]

snow = SnowballStemmer('english')
stop = set(stopwords.words('english'))

#spits out a 'cleaned' text indexed after the location of a string, if there's any.
def truncStrings(text, string, index):
    try:
        j = text.index(str(string))
        newstrings = text[j + len(string):]
        return truncStrings(newstrings, string, j)
    except ValueError:
        # print("the " + string + "was found in index number: " + str(index))
        return text

#unifies stemmed words and stop words within the string
def stopstem(string):
    #tokenize the words
    words = string.split()
    #removes stop words
    wordlist = [i for i in words if i not in stop]
    #unifies stemmed words
    wordlist = [snow.stem(i) for i in wordlist]
    return ' '.join(wordlist)

#removes unnecessary punctuation
def rmpunctuation(string):
    string = re.sub(r'\\', "", string)
    string = re.sub(r'-', ' ', string)
    return ' '.join(word.strip(st.punctuation) for word in string.split())

#cleans the data
def clean(hrc_data, remove):
    hrc_copy = hdrc_data.copy()
    for i in range(len(hrc_copy)):
        #truncates string before "subjects:", "sent:", "re:", and "fw:" to display only the latest content
        newtext = truncStrings(hrc_copy.loc[i, "text"], "subject:", 0)
        newtext = truncStrings(newtext, "sent:", 0)
        newtext = truncStrings(newtext, "re:", 0)
        newtext = truncStrings(newtext, "fw:", 0)
        for bad in badstrings:
            newtext = re.sub(bad, "", newtext)
        #removes stop words and unifies stemmed words
        newtext = stopstem(newtext)
        newtext = rmpunctuation(newtext)
        hrc_copy.loc[i, "text"] = newtext
    return hrc_copy


new = clean(hrc_train, badstrings)