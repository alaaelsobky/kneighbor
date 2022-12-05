import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
#library(imbalance)
df = pd.read_csv('abalone1.csv')
#print(df['Outcome'].value_counts())
#count_classes = pd.value_counts(df['Outcome'], sort = True)
#count_classes.plot(kind = 'bar', rot=0)
#plt.title("Class Distribution")
#plt.xlabel("Class")
#plt.ylabel("Frequency")
#plt.show()
#print(imbalanceRatio(df, classAttr = "Sex"))
X = df.drop('Sex',axis = 1)
Y = df['Sex']
#print(X)
#print(Y)

lr =KNeighborsClassifier(n_neighbors = 5)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

# train the model on train set
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
# print classification report
print('Presampled dataset shape {}'.format(Counter(Y)))
#print(classification_report(Y_test, predictions))

print(accuracy_score(Y_test, predictions))
#under_sampling
nm = NearMiss()
X_res,y_res=nm.fit_resample(X,Y)
print('Resampled NearMiss dataset shape {}'.format(Counter(y_res)))

X_train,X_test,Y_train,Y_test = train_test_split(X_res,y_res,test_size=0.3)

# train the model on train set
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
# print classification report
#print(classification_report(Y_test, predictions))
print(accuracy_score(Y_test, predictions))
#auc = metrics.roc_auc_score(y_test, predictions)
#print('aaa')
#print AUC score
#print(auc)


nm = RandomUnderSampler()
X_res,y_res=nm.fit_resample(X,Y)
print('Resampled RandomUnderSampler dataset shape {}'.format(Counter(y_res)))

X_train,X_test,Y_train,Y_test = train_test_split(X_res,y_res,test_size=0.3)
# train the model on train set
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
# print classification report
#print(classification_report(Y_test, predictions))
print(accuracy_score(Y_test, predictions))



#over_sampling
os =RandomOverSampler()
X_res, y_res = os.fit_resample(X, Y)
print('Resampled RandomOverSampler dataset shape {}'.format(Counter(y_res)))

X_train,X_test,Y_train,Y_test = train_test_split(X_res,y_res,test_size=0.3)
# logistic regression object
#lr = LogisticRegression(max_iter=10000,solver='lbfgs')
# train the model on train set
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
# print classification report
#print(classification_report(Y_test, predictions))
print(accuracy_score(Y_test, predictions))

os=SMOTE()
X_res, y_res = os.fit_resample(X, Y)
print('Resampled SMOTE dataset shape {}'.format(Counter(y_res)))

X_train,X_test,Y_train,Y_test = train_test_split(X_res,y_res,test_size=0.3)
# logistic regression object
#lr = LogisticRegression(max_iter=10000,solver='lbfgs')
# train the model on train set
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
# print classification report
#print(classification_report(Y_test, predictions))
print(accuracy_score(Y_test, predictions))

#over_sampling & over_sampling
smk = SMOTETomek()
X_res,y_res=smk.fit_resample(X,Y)
print('Resampled SMOTETomek dataset shape {}'.format(Counter(y_res)))

X_train,X_test,Y_train,Y_test = train_test_split(X_res,y_res,test_size=0.3)
# logistic regression object
#lr = LogisticRegression(max_iter=10000,solver='lbfgs')
# train the model on train set
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
# print classification report
#print(classification_report(Y_test, predictions))
print(accuracy_score(Y_test, predictions))


smk = SMOTEENN()
X_res,y_res=smk.fit_resample(X,Y)
print('Resampled SMOTEENN dataset shape {}'.format(Counter(y_res)))

X_train,X_test,Y_train,Y_test = train_test_split(X_res,y_res,test_size=0.3)
# logistic regression object
#lr = LogisticRegression(max_iter=10000,solver='lbfgs')
# train the model on train set
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
# print classification report
#print(classification_report(Y_test, predictions))
print(accuracy_score(Y_test, predictions))

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
brf = BalancedRandomForestClassifier()
brf.fit(X_train,Y_train)
#print(brf.score(X_train,Y_train))


