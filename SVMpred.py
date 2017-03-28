import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics, svm
from sklearn import model_selection
import matplotlib.pyplot as plt


v = CountVectorizer(ngram_range=(1, 2), stop_words="english")

#retrives the Cross-validation value for data
def getCV(estimator, data, labels):
    return np.mean(model_selection.cross_val_score(estimator, data, labels, cv = 5))

#sets the margin for loops in getBestSVMParam function
def getSVMEst(c):
    return svm.LinearSVC(C = c)

#retrives the best margin value 'c' for the data, and plot the data
# def getBestSVMParam(data, labels, cmin = 0.01, cmax = 300):
#     minCV = float('inf')
#     minParam = 0
#     costs = []
#     cvs = []
#     for c in np.arange(cmin, cmax, 0.1):
#         svms = getSVMEst(c)
#         cv = getCV(svms, data, labels)
#         if cv < minCV:
#             minCV = cv
#             minParam = c
#         costs.append(c)
#         cvs.append(cv)
#     plt.plot(costs, cvs)
#     plt.xlabel("costs")
#     plt.ylabel("cross-validation scores")
#     plt.show()
#     return minParam

# set path
train_path, test_path = "data/HRC_train.tsv", "data/HRC_test.tsv"
# import train data & set variables
email_train = pd.read_table(train_path, header=None, names=['id', 'txt'], sep='\t')
y_train, X_train = email_train.id, email_train.txt
# import test data & set variables
email_test = pd.read_table(test_path, header=None, names=['id', 'txt'], sep='\t')
y_test, X_test = email_test.id, email_test.txt

#get labels for cross-validation
label = email_train['id'].values
# make doc-term matrices
X_train_dtm = v.fit_transform(X_train.values.astype('U'))
X_test_dtm = v.transform(X_test.values.astype('U'))

# bestparam = getBestSVMParam(X_train_dtm, label)

#print accuracy
clf = svm.LinearSVC()
y_pred_class = clf.fit(X_train_dtm, y_train).predict(X_test_dtm)
print("accuracy was ", metrics.accuracy_score(y_test, y_pred_class))
