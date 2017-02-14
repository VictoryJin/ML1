# Email Text Classification

### Contents

[Building the Package](#bulding-the-package)  
[Project Info](#project-info)



## Building the Package
I would suggest two ways to set up packages, both which require the latest version of [Python3](https://www.python.org/downloads/) and  [Anaconda](https://www.continuum.io/downloads). You can either choose to install python packages individually, or use the predefined virtual environment provided in the files.


### Predefined Environment
Once you have both anaconda and python installed, navigate to where you've cloned the files and run the following code:
```
conda env create -f environment.yaml
```
This will create a virtual environment called *ml_env*, with all the necessary packages installed to run the files.
To run the environment, from terminal run:
```python
source activate ml_env      #on OSX/Linux
activate ml_env             #on Windows
```
and to deactivate, run:
```python
source deactivate           #on OSX/Linux
deactivate             	    #on Windows
```
If you want to remove the environment,  run:
```python
conda env remove -n ml_env
```
and this will remove the specified environment *ml_env*.


### Installing Individual Packages
You can also start from any Python3 distribution but you need to install the following libraries in order to run the files.
* NumPy & Pandas
```
conda install numpy pandas
```
* nltk
```
conda install nltk
```
* scikit-learn
```
conda install scikit-learn
```
* Matplotlib & Seaborn
```
conda install matplotlib seaborn
```
## Project Info
### Introduction
**Machine Learning to predict email contacts for the once controversial HRC emails.**  
This is my improved version (from scratch) of what was a term-project with a group of 4 in a ML class in UC Berkeley.  The link to the original project can be found [here](https://github.com/liyu1390/STAT154-GROUP08).  
The goal of this project is to identify the right parameters and 'power features' to correctly identify the senders of 389 emails by training from 3505 emails.  
My aims for this project is to fully understand for myself the ML technique, concepts and implementations.
Some improvements I hope to achieve are but not limited to:

* Increase test data accuracy (compared to 74% accuracy for the group project).
* Clean and optimize codes for readabilty.
* Visualize the results using *matplotlib* and *seaborn*.



### Process
1. Raw data is cleaned through custom parameters using the *nltk library* to remove stop words, unify stemmed words,
and remove "bad strings" that are used for every email.
2. The cleaned data is then output in the *data* folder, which I will use for the algorithms.
3. Use *CountVectorizer()* from *sklearn.feature_extraction.text* to transform the data into a doc-term matrix.
4. Test for the necessary parameters of each algorithm.
5. Fit the model using the parameters and the train set to predict the test set.
6. Measure the accuracy of the prediction with the actual values of the test set.



### Files
**Currently coding for LDA & SVM for text classification**
* test_cleaner.py = cleans the strings with stop/stem words, universal strings, and punctuations(finished) and outputs the cleaned indexed file when run.  
*Do NOT need to run again if HRC_clean.tsv is in ./data/*
* data_explorers.py = code for myself to explore the data and identify patterns.
* data_explorers.ipiynb = same as data_explorers.py, but for Jupyter notebook.
* LDA.py = outputs the prediction accuracy using Linear Discriminant Analysis with singular value decomposition.  
*WARNING: the current code takes up excessive memory, and can terminate the process if the computer has low memory*
* SVMpred.py = outputs the prediction accuracy using SVM (work in progress)
* NBpred.py = outputs the prediction accuracy using Naive Bayes.



### References
[sklearn - Preprocessing](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning)  
[sklearn - Supervised Learning](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning)  
[sklearn - Clustering](http://scikit-learn.org/stable/modules/clustering.html#clustering)  
[Udacity.com](http://www.udacity.com)

