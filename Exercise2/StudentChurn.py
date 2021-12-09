# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np

col_names = ["Id", "Churn", "Line", "Grade", "Age", "Distance", "StudyGroup"]

data = pd.read_csv('StudentChurn.csv', sep=';')

# show the data
print(data.describe(include='all'))
# the describe method is a great way to get an overview of the data
print(data.values)
print(data.columns)
print(data.shape)

data.dropna(axis=0, inplace=True)
data.drop(columns='Id', axis=1, inplace=True)


data.replace({'Line': {'HTX': 1.0, 'HF': 2.0, 'STX': 3.0, 'HHX': 4.0, 'EUX': 5.0},
              'Churn': {'Completed': 0, 'Stopped': 1}}, inplace=True)
print(data.isnull().sum())

yvalues = pd.DataFrame(dict(Churn=[]), dtype=int)
yvalues['Churn'] = data['Churn'].copy()
data.drop('Churn', axis=1, inplace=True)

# split the data
X_train, X_test, Y_train, Y_test = train_test_split(data, yvalues, test_size=0.2)

# scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Neural net classification, predictions and classification report
mlp = MLPClassifier(hidden_layer_sizes=(16), max_iter=10000, random_state=0)
mlp.fit(X_train, Y_train.values.ravel())
mlpPredictions = mlp.predict(X_test)
print('-- Neural Net --')
print(classification_report(Y_test, mlpPredictions, target_names=['Completed', 'Stopped']))

# SVM
random_grid = {'C': [int(x) for x in np.linspace(start=50, stop=400, num=10)],
               'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
               'degree': [int(x) for x in np.linspace(start=3, stop=20, num=10)],
               'gamma': [float(x) for x in np.linspace(start=0.1, stop=1, num=10)]}
print(random_grid)
svc = SVC(random_state=1)  # probability=True
svc_random = RandomizedSearchCV(svc, param_distributions=random_grid, n_iter=10, verbose=1,
                                cv=2, n_jobs=4, random_state=42)
svc_random.fit(X_train, Y_train.values.ravel())
print(svc_random.best_estimator_)
svc_predictions = svc_random.best_estimator_.predict(X_test)

print('-- SVC --')
print(classification_report(Y_test, svc_predictions, target_names=['Completed', 'Stopped']))
