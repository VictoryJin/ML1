import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics, svm

v = CountVectorizer()
clf = svm.LinearSVC()

# set path
train_path, test_path = "data/HRC_train_cleaned.tsv", "data/HRC_test_cleaned.tsv"
# import train data & set variables
email_train = pd.read_table(train_path, header=None, names=['id', 'txt'], sep='\t')
y_train, X_train = email_train.id, email_train.txt
# import test data & set variables
email_test = pd.read_table(test_path, header=None, names=['id', 'txt'], sep='\t')
y_test, X_test = email_test.id, email_test.txt

# make doc-term matrices
X_train_dtm = v.fit_transform(X_train.values.astype('U'))
X_test_dtm = v.transform(X_test.values.astype('U'))

y_pred_class = clf.fit(X_train_dtm, y_train).predict(X_test_dtm)
print("accuracy was ", metrics.accuracy_score(y_test, y_pred_class))
