import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

vect = CountVectorizer()
path1 = "data/HRC_train_cleaned.tsv"
path2 = "data/HRC_test_cleaned.tsv"
nb = MultinomialNB()
nb1 = MultinomialNB()
email_train = pd.read_table(path1, header = None, names = ['id', 'txt'], sep = '\t')
email_test = pd.read_table(path2, header = None, names = ['id', 'txt'], sep = '\t')

X_train = email_train.txt
y_train = email_train.id
X_test = email_test.txt
y_test = email_test.id

X_train_dtm = vect.fit_transform(X_train.values.astype('U'))
X_test_dtm = vect.transform(X_test.values.astype('U'))
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)
print("accuracy was ", metrics.accuracy_score(y_test, y_pred_class))
print(metrics.confusion_matrix(y_test, y_pred_class))

path3 = "data/HRC_train.tsv"
path4 = "data/HRC_test.tsv"
email_train1 = pd.read_table(path3, header = None, names = ['id', 'txt'], sep = '\t')
email_test1 = pd.read_table(path4, header = None, names = ['id', 'txt'], sep = '\t')

X_train1 = email_train1.txt
y_train1 = email_train1.id
X_test1 = email_test1.txt
y_test1 = email_test1.id

X_train_dtm1 = vect.fit_transform(X_train1)
X_test_dtm1 = vect.transform(X_test1)
nb1.fit(X_train_dtm1, y_train1)
y_pred_class1 = nb1.predict(X_test_dtm1)
print("accuracy was ", metrics.accuracy_score(y_test1, y_pred_class1))
print(metrics.confusion_matrix(y_test1, y_pred_class1))