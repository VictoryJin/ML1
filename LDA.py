import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics

from scipy.sparse import csr_matrix

print("WARNING: running LDA with dimension reduction would take some time.\n"
      "Check if the shrinkage = 'none' to save time. \n"
      "otherwise I'd recommend a good RAM and CPU, or AWS :)")

v = CountVectorizer()
lda = LinearDiscriminantAnalysis(solver = "svd")


#set path
train_path, test_path = "data/HRC_train_cleaned.tsv", "data/HRC_test_cleaned.tsv"
#import train data & set variables
email_train = pd.read_table(train_path, header = None, names = ['id', 'txt'], sep = '\t')
y_train, X_train = email_train.id, email_train.txt
#import test data & set variables
email_test = pd.read_table(test_path, header = None, names = ['id', 'txt'], sep = '\t')
y_test, X_test = email_test.id, email_test.txt


#make doc-term matrices
X_train_dtm = v.fit_transform(X_train.values.astype('U')).toarray()
X_test_dtm = v.transform(X_test.values.astype('U')).toarray()
#fit and predict wrt LDA
y_pred_class = lda.fit(X_train_dtm, y_train)
y_pred_class


print("Model had", X_train_dtm.shape[1], "number of features, with", X_train_dtm.shape[0],
      "train observations and", X_test_dtm.shape[0], "test observations.")
print("The accuracy of the model onto the test set was", metrics.accuracy_score(y_test, y_pred_class))
print(metrics.confusion_matrix(y_test, y_pred_class))


# LDA analysis without shrinkage (with svd) had an accuracy of 0.36503856




