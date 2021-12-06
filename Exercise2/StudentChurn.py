# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:41:07 2021

@author: sila
"""

# import panda library and a few others we will need.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

col_names = ["Id", "Churn", "Line", "Grade", "Age", "Distance", "StudyGroup"]

data = pd.read_csv('StudentChurn.csv', sep=';')

# show the data
print(data.describe(include='all'))

data.dropna(axis=0, inplace=True)

data.drop(columns='Id', axis=1, inplace=True)

# data['Churn'] = data['Churn'].replace(['Completed'], 1.0)
data.replace({'Churn': {'Completed': 0.0, 'Stopped': 1.0},
              'Line': {'HTX': 1.0, 'HF': 2.0, 'STX': 3.0, 'HHX': 4.0, 'EUX': 5.0}}, inplace=True)
print(data.isnull().sum())

yvalues = pd.DataFrame(dict(Churn=[]), dtype=int)
yvalues['Churn'] = data['Churn'].copy()
data.drop('Churn', axis=1, inplace=True)

X_train, X_test, Y_train, Y_test = train_test_split(data, yvalues, test_size=0.2)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# Neural net
mlp = MLPClassifier(hidden_layer_sizes=(16), max_iter=10000, random_state=0)
mlp.fit(X_train, Y_train.values.ravel())
mlpPredictions = mlp.predict(X_test)
print('-- Neural Net --')
print(classification_report(Y_test, mlpPredictions, target_names=['Completed', 'Stopped']))

# Decision Tree
decision_tree = DecisionTreeClassifier(max_depth=3)
decision_tree.fit(X_train, Y_train)
decision_tree_predictions = decision_tree.predict(X_test)
print('-- Decision Tree --')
print(classification_report(Y_test, decision_tree_predictions, target_names=['Completed', 'Stopped']))
