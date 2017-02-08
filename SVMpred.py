import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics, svm

vect = CountVectorizer()
path1 = "data/HRC_train_cleaned.tsv"
path2 = "data/HRC_test_cleaned.tsv"
email_train = pd.read_table(path1, header = None, names = ['id', 'txt'], sep = '\t')
email_test = pd.read_table(path2, header = None, names = ['id', 'txt'], sep = '\t')

X_train = email_train.txt
y_train = email_train.id
X_test = email_test.txt
y_test = email_test.id
X_train_dtm = vect.fit_transform(X_train.values.astype('U'))
X_test_dtm = vect.transform(X_test.values.astype('U'))


lin_clf = svm.LinearSVC()
lin_clf.fit(X_train_dtm, y_train)
y_pred_class = lin_clf.predict(X_test_dtm)
print("accuracy was ", metrics.accuracy_score(y_test, y_pred_class))