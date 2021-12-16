# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
import numpy as np
import pydotplus
import matplotlib.image as mpimg
import io

col_names = ["Id", "Churn", "Line", "Grade", "Age", "Distance", "StudyGroup"]

data = pd.read_csv('StudentChurn.csv', sep=';')

# show the data
print(data.describe(include='all'))
# the describe method is a great way to get an overview of the data
print(data.values)
print(data.columns)
print(data.shape)

data.dropna(axis=0, inplace=True)
# data.fillna(data.mode().iloc[0], inplace=True)
data.drop(columns='Id', axis=1, inplace=True)

data.replace({'Line': {'HTX': 1.0, 'HF': 2.0, 'STX': 3.0, 'HHX': 4.0, 'EUX': 5.0},
              'Churn': {'Completed': 1, 'Stopped': 0}}, inplace=True)
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
# mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 8), max_iter=10000, random_state=0)
# mlp.fit(X_train, Y_train.values.ravel())
# mlpPredictions = mlp.predict(X_test)
# print('-- Neural Net --')
# print(classification_report(Y_test, mlpPredictions, target_names=['Completed', 'Stopped']))
nn_best_acc = {'acc': 0, 'neurons_i': 0}
mlp_rep = None

for i in range(15, 25):
    mlp = MLPClassifier(hidden_layer_sizes=(i), max_iter=10000)
    mlp.fit(X_train, Y_train.values.ravel())

    mlp_predictions = mlp.predict(X_test)
    acc = accuracy_score(Y_test, mlp_predictions)
    if acc > nn_best_acc['acc']:
        nn_best_acc['acc'] = acc
        nn_best_acc['neurons_i'] = mlp.hidden_layer_sizes
        mlp_rep = classification_report(Y_test, mlp_predictions, target_names=['Stopped', 'Completed'])
        matrix = confusion_matrix(Y_test, mlp_predictions)

print(f'best acc: {nn_best_acc["acc"]} at {nn_best_acc["neurons_i"]} neurons')
print('Neural net \n', mlp_rep)
tn, fp, fn, tp = matrix.ravel()
print(f'Recall rate: {(tp / (tp + fn)):.2f}\n'
      f'Precision rate: {(tp / (tp + fp)):.2f}')

# --- Decision Tree ---
dtc = DecisionTreeClassifier(max_depth=10)

dtc.fit(X_train, Y_train.values.ravel())
dtc_predictions = dtc.predict(X_test)
print('Decision Tree \n', classification_report(Y_test, dtc_predictions, target_names=['Stopped', 'Completed']))
tn, fp, fn, tp = confusion_matrix(Y_test, dtc_predictions).ravel()
print(f'Recall rate: {(tp / (tp + fn)):.2f}\n'
      f'Precision rate: {(tp / (tp + fp)):.2f}')

target_names = ['Completed', 'Stopped']
# for name, score in zip(data.columns[0:5], dtc.feature_importances_):
#     print("feature importance: ", name, score)
#
# dot_data = io.StringIO()
# export_graphviz(dtc,
#                 out_file=dot_data,
#                 feature_names=data.columns[0:],
#                 class_names=target_names,
#                 rounded=True,
#                 filled=True)
#
# filename = "tree.png"
# pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png(filename)  # write the dot data to a pgn file
# img = mpimg.imread(filename)  # read this pgn file
#
# plt.figure(figsize=(8, 8))  # setting the size to 10 x 10 inches of the figure.
# imgplot = plt.imshow(img)  # plot the image.
# plt.show()

grades = ['02-4', '4-7', '7-10', '10-12']
stopped = data.loc[yvalues['Churn'] == 0]
completed = data.loc[yvalues['Churn'] == 1]

grades_2_4_stopped = stopped.loc[stopped['Grade'] < 4]
grades_4_7_stopped = stopped.loc[(stopped['Grade'] >= 4) & (stopped['Grade'] < 7)]
grades_7_10_stopped = stopped.loc[(stopped['Grade'] >= 7) & (stopped['Grade'] < 10)]
grades_10_12_stopped = stopped.loc[(stopped['Grade'] >= 10) & (stopped['Grade'] < 12)]

grades_stopped = [
    len(stopped.loc[stopped['Grade'] < 4]),
    len(stopped.loc[(stopped['Grade'] >= 4) & (stopped['Grade'] < 7)]),
    len(stopped.loc[(stopped['Grade'] >= 7) & (stopped['Grade'] < 10)]),
    len(stopped.loc[(stopped['Grade'] >= 10) & (stopped['Grade'] < 12)])
]
grades_completed = [
    len(completed.loc[completed['Grade'] < 4]),
    len(completed.loc[(completed['Grade'] >= 4) & (completed['Grade'] < 7)]),
    len(completed.loc[(completed['Grade'] >= 7) & (completed['Grade'] < 10)]),
    len(completed.loc[(completed['Grade'] >= 10) & (completed['Grade'] < 12)]),
]
print(grades_stopped, '\n', grades_completed)

X_axis = np.arange(len(grades))
print(X_axis)

plt.bar(X_axis - 0.2, grades_completed, 0.4, label = 'completed')
plt.bar(X_axis + 0.2, grades_stopped, 0.4, label = 'stopped')

plt.xticks(X_axis, grades)
plt.ylabel('Number of student')
plt.xlabel('Grade groups')
plt.title('Student who completed and stopped for each grade group')
plt.legend()
plt.show()

