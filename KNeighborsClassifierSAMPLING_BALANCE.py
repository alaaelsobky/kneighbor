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
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import metrics

df = pd.read_csv('balance.csv')
# print(df['Outcome'].value_counts())
# count_classes = pd.value_counts(df['Outcome'], sort = True)
# count_classes.plot(kind = 'bar', rot=0)
# plt.title("Class Distribution")
# plt.xlabel("Class")
# plt.ylabel("Frequency")
# plt.show()

X = df.drop('Class', axis=1)
Y = df['Class']
# print(X)
# print(Y)
lr = KNeighborsClassifier(n_neighbors=5)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# print out splits
print('Class B        :', round(Counter(Y_test)['B'] / len(Y_test) * 100, 2), '% of the dataset')
print('Class L        :', round(Counter(Y_test)['L'] / len(Y_test) * 100, 2), '% of the dataset')
print('Class R        :', round(Counter(Y_test)['R'] / len(Y_test) * 100, 2), '% of the dataset')

# train the model on train set
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
print('Resampled dataset shape {}'.format(Counter(Y)))
# print classification report
# print(classification_report(Y_test, predictions,zero_division=1))
# print(f1_score(Y_test, predictions, average='macro'))
# print(precision_recall_fscore_support(Y_test, predictions, average='macro'))
# print(accuracy_score(Y_test, predictions))
#print(metrics.classification_report(Y_test, predictions))
cv = KFold(n_splits=5, shuffle=True, random_state=42)
f1_score =  cross_val_score(lr, X, Y, cv=cv, scoring='f1_macro').mean()
recall_score = cross_val_score(lr, X, Y, cv=cv,  scoring='recall_macro').mean()
precision = cross_val_score(lr, X, Y, cv=cv,  scoring='precision_macro').mean()
accuracy =  cross_val_score(lr, X, Y, cv=cv, scoring='accuracy').mean()
print("Mean accuracy: %0.2f " % (accuracy) )
print("Mean F1: %0.2f " % (f1_score) )
print("Mean recall: %0.2f " % (recall_score) )
print("Mean precision: %0.2f " % (precision) )

print('')
# under_sampling
nm = NearMiss()
X_res, y_res = nm.fit_resample(X, Y)
print('Resampled NearMiss dataset shape {}'.format(Counter(y_res)))

#X_train, X_test, Y_train, Y_test = train_test_split(X_res, y_res, test_size=0.3)
# print(Counter(y_res)['B']/len(y_res))
# train the model on train set
#lr.fit(X_train, Y_train)
#predictions = lr.predict(X_test)
# print classification report
# print(classification_report(Y_test, predictions,zero_division=1))
# print(f1_score(Y_test, predictions, average='macro'))
#print(precision_recall_fscore_support(Y_test, predictions, average='macro'))
# print(accuracy_score(Y_test, predictions))
f1_score =  cross_val_score(lr, X_res, y_res, cv=cv, scoring='f1_macro').mean()
recall_score = cross_val_score(lr, X_res, y_res, cv=cv,  scoring='recall_macro').mean()
precision = cross_val_score(lr, X_res, y_res, cv=cv,  scoring='precision_macro').mean()
accuracy =  cross_val_score(lr,  X_res, y_res, cv=cv, scoring='accuracy').mean()
print("Mean accuracy: %0.2f " % (accuracy) )
print("Mean F1: %0.2f " % (f1_score) )
print("Mean recall: %0.2f " % (recall_score) )
print("Mean precision: %0.2f " % (precision) )

nm = RandomUnderSampler()
X_res, y_res = nm.fit_resample(X, Y)
print('Resampled RandomUnderSampler dataset shape {}'.format(Counter(y_res)))

#X_train, X_test, Y_train, Y_test = train_test_split(X_res, y_res, test_size=0.3)
# train the model on train set
#lr.fit(X_train, Y_train)
#predictions = lr.predict(X_test)
# print classification report
# print(classification_report(Y_test, predictions,zero_division=1))
# print(f1_score(Y_test, predictions, average='macro'))
#print(precision_recall_fscore_support(Y_test, predictions, average='macro'))
# print(accuracy_score(Y_test, predictions))
f1_score =  cross_val_score(lr, X_res, y_res, cv=cv, scoring='f1_macro').mean()
recall_score = cross_val_score(lr, X_res, y_res, cv=cv,  scoring='recall_macro').mean()
precision = cross_val_score(lr, X_res, y_res, cv=cv,  scoring='precision_macro').mean()
accuracy =  cross_val_score(lr,  X_res, y_res, cv=cv, scoring='accuracy').mean()
print("Mean accuracy: %0.2f " % (accuracy) )
print("Mean F1: %0.2f " % (f1_score) )
print("Mean recall: %0.2f " % (recall_score) )
print("Mean precision: %0.2f " % (precision) )

# over_sampling
os = RandomOverSampler()
X_res, y_res = os.fit_resample(X, Y)
print('Resampled RandomOverSampler dataset shape {}'.format(Counter(y_res)))
#X_train, X_test, Y_train, Y_test = train_test_split(X_res, y_res, test_size=0.3)
# train the model on train set
#lr.fit(X_train, Y_train)
#predictions = lr.predict(X_test)
# print classification report
# print(classification_report(Y_test, predictions,zero_division=1))
# print(f1_score(Y_test, predictions, average='macro'))
#print(precision_recall_fscore_support(Y_test, predictions, average='macro'))
# print(accuracy_score(Y_test, predictions))
f1_score =  cross_val_score(lr, X_res, y_res, cv=cv, scoring='f1_macro').mean()
recall_score = cross_val_score(lr, X_res, y_res, cv=cv,  scoring='recall_macro').mean()
precision = cross_val_score(lr, X_res, y_res, cv=cv,  scoring='precision_macro').mean()
accuracy =  cross_val_score(lr,  X_res, y_res, cv=cv, scoring='accuracy').mean()
print("Mean accuracy: %0.2f " % (accuracy) )
print("Mean F1: %0.2f " % (f1_score) )
print("Mean recall: %0.2f " % (recall_score) )
print("Mean precision: %0.2f " % (precision) )

os = SMOTE()
X_res, y_res = os.fit_resample(X, Y)
print('Resampled SMOTE dataset shape {}'.format(Counter(y_res)))
#X_train, X_test, Y_train, Y_test = train_test_split(X_res, y_res, test_size=0.3)
# train the model on train set
#lr.fit(X_train, Y_train)
#predictions = lr.predict(X_test)
# print classification report
# print(classification_report(Y_test, predictions,zero_division=1))
# print(f1_score(Y_test, predictions, average='macro'))
#print(precision_recall_fscore_support(Y_test, predictions, average='macro'))
# print(accuracy_score(Y_test, predictions))
f1_score =  cross_val_score(lr, X_res, y_res, cv=cv, scoring='f1_macro').mean()
recall_score = cross_val_score(lr, X_res, y_res, cv=cv,  scoring='recall_macro').mean()
precision = cross_val_score(lr, X_res, y_res, cv=cv,  scoring='precision_macro').mean()
accuracy =  cross_val_score(lr,  X_res, y_res, cv=cv, scoring='accuracy').mean()
print("Mean accuracy: %0.2f " % (accuracy) )
print("Mean F1: %0.2f " % (f1_score) )
print("Mean recall: %0.2f " % (recall_score) )
print("Mean precision: %0.2f " % (precision) )

# over_sampling & over_sampling
smk = SMOTETomek()
X_res, y_res = smk.fit_resample(X, Y)
print('Resampled SMOTETomek dataset shape {}'.format(Counter(y_res)))
#X_train, X_test, Y_train, Y_test = train_test_split(X_res, y_res, test_size=0.3)
# train the model on train set
#lr.fit(X_train, Y_train)
#predictions = lr.predict(X_test)
# print classification report
# print(classification_report(Y_test, predictions,zero_division=1))
# print(f1_score(Y_test, predictions, average='macro'))
#print(precision_recall_fscore_support(Y_test, predictions, average='macro'))
# print(accuracy_score(Y_test, predictions))
f1_score =  cross_val_score(lr, X_res, y_res, cv=cv, scoring='f1_macro').mean()
recall_score = cross_val_score(lr, X_res, y_res, cv=cv,  scoring='recall_macro').mean()
precision = cross_val_score(lr, X_res, y_res, cv=cv,  scoring='precision_macro').mean()
accuracy =  cross_val_score(lr,  X_res, y_res, cv=cv, scoring='accuracy').mean()
print("Mean accuracy: %0.2f " % (accuracy) )
print("Mean F1: %0.2f " % (f1_score) )
print("Mean recall: %0.2f " % (recall_score) )
print("Mean precision: %0.2f " % (precision) )


smk = SMOTEENN()
X_res, y_res = smk.fit_resample(X, Y)
print('Resampled SMOTEENN dataset shape {}'.format(Counter(y_res)))
#X_train, X_test, Y_train, Y_test = train_test_split(X_res, y_res, test_size=0.3)
# train the model on train set
#lr.fit(X_train, Y_train)
#predictions = lr.predict(X_test)
# print classification report
# print(classification_report(Y_test, predictions,zero_division=1))
# print(f1_score(Y_test, predictions, average='macro'))
#print(precision_recall_fscore_support(Y_test, predictions, average='macro'))
# print(accuracy_score(Y_test, predictions))
f1_score =  cross_val_score(lr, X_res, y_res, cv=cv, scoring='f1_macro').mean()
recall_score = cross_val_score(lr, X_res, y_res, cv=cv,  scoring='recall_macro').mean()
precision = cross_val_score(lr, X_res, y_res, cv=cv,  scoring='precision_macro').mean()
accuracy =  cross_val_score(lr,  X_res, y_res, cv=cv, scoring='accuracy').mean()
print("Mean accuracy: %0.2f " % (accuracy) )
print("Mean F1: %0.2f " % (f1_score) )
print("Mean recall: %0.2f " % (recall_score) )
print("Mean precision: %0.2f " % (precision) )

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
brf = BalancedRandomForestClassifier()
brf.fit(X_train, Y_train)
# print(brf.score(X_train,Y_train))


