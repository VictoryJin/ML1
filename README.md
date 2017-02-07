# ML1
Machine Learning to predict email contacts. <br />
This is my improved version (from scratch) of what was a term-project with a group of 4 in a ML class. <br />
The goal of this project is to fully understand for myself the ML technique, concepts and implementations <br />
Some improvements I hope to achieve are but not limited to:
* Add PCA to reduce the dimensions of an otherwise 3505x60,000 data frame, which we did with brute force on our first attempt project<br />
* Increase test data accuracy
* Clean and optimize codes <br />

The files are explained as follows: <br />
1. cleaner.py = cleans the strings with stop/stem words, universal strings, and punctuations(finished) and outputs the cleaned indexed file when run. *Do NOT need to run again if HRC_clean.tsv is in ./data/* <br />
2. data_explorers.py = code for myself to explore the data and identify patterns <br />
3. data_explorers.ipiynb = same as data_explorers.py, but for Jupyter notebook. <br />
**Improved versions of SVM, RF, and K-means algorithms coming soon!**
