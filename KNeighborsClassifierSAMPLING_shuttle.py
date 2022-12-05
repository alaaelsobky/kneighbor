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
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from sklearn.svm import LinearSVC
RANDOM_STATE = 42
df = pd.read_csv('shuttle.csv')
#print(df['Outcome'].value_counts())
#count_classes = pd.value_counts(df['Outcome'], sort = True)
#count_classes.plot(kind = 'bar', rot=0)
#plt.title("Class Distribution")
#plt.xlabel("Class")
#plt.ylabel("Frequency")
#plt.show()

X = df.drop('class',axis = 1)
Y = df['class']
#print(X)
#print(Y)

lr =KNeighborsClassifier(n_neighbors = 2)
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

# train the model on train set
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
print('Resampled dataset shape {}'.format(Counter(Y)))
# print classification report
print(classification_report_imbalanced(Y_test, predictions,zero_division=1))
#print(classification_report(Y_test, predictions,zero_division=1))
#print(accuracy_score(Y_test, predictions))


#under_sampling
print('Resampled NearMiss dataset shape {}'.format(Counter(Y)))
# Create a pipeline
pipeline = make_pipeline(NearMiss(version=2),KNeighborsClassifier(n_neighbors = 2))
pipeline.fit(X_train, Y_train)

# Classify and report the results
print(classification_report_imbalanced(Y_test, pipeline.predict(X_test),zero_division=1))
# print classification report
#print(classification_report(Y_test, predictions))


print('Resampled RandomUnderSampler dataset shape {}'.format(Counter(Y)))

# Create a pipeline
pipeline = make_pipeline(RandomUnderSampler(),KNeighborsClassifier(n_neighbors = 2))
pipeline.fit(X_train, Y_train)

# Classify and report the results
print(classification_report_imbalanced(Y_test, pipeline.predict(X_test),zero_division=1))



#over_sampling

# Create a pipeline
pipeline = make_pipeline(RandomOverSampler(),KNeighborsClassifier(n_neighbors = 2))
pipeline.fit(X_train, Y_train)

# Classify and report the results
print(classification_report_imbalanced(Y_test, pipeline.predict(X_test),zero_division=1))

print('Resampled SMOTE dataset shape {}'.format(Counter(Y)))

# Create a pipeline

strategy = { 1:1700, 2:1700, 3:1700, 4:1700, 5:1700}
pipeline = make_pipeline(SMOTE(sampling_strategy=strategy),KNeighborsClassifier(n_neighbors = 2))
pipeline.fit(X_train, Y_train)

# Classify and report the results
print(classification_report_imbalanced(Y_test, pipeline.predict(X_test),zero_division=1))

#over_sampling & over_sampling
print('Resampled SMOTETomek dataset shape {}'.format(Counter(Y)))
# Create a pipeline
pipeline = make_pipeline(SMOTETomek(),KNeighborsClassifier(n_neighbors = 2))
pipeline.fit(X_train, Y_train)

# Classify and report the results
print(classification_report_imbalanced(Y_test, pipeline.predict(X_test),zero_division=1))

print('Resampled SMOTEENN dataset shape {}'.format(Counter(Y)))
# Create a pipeline
pipeline = make_pipeline(SMOTEENN(),KNeighborsClassifier(n_neighbors = 2))
pipeline.fit(X_train, Y_train)

# Classify and report the results
print(classification_report_imbalanced(Y_test, pipeline.predict(X_test),zero_division=1))


