# -*- coding: utf-8 -*-

from sklearn.datasets import load_digits
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

mnist = load_digits()

print("Data Shape:")
print(pd.DataFrame(mnist.data).shape)

print("Data Head:")
print(pd.DataFrame(mnist.data).head())

print("Target Shape:")
print(pd.DataFrame(mnist.target).shape)

fig, axes = plt.subplots(2, 10, figsize=(16, 6))
for i in range(20):
    axes[i // 10, i % 10].imshow(mnist.images[i], cmap='gray')
    axes[i // 10, i % 10].axis('off')
    axes[i // 10, i % 10].set_title(f"target: {mnist.target[i]}")

plt.tight_layout()

plt.show()

X_train, X_test, y_train, y_test = train_test_split(mnist.data,
                                                    mnist.target,
                                                    test_size=0.2,
                                                    random_state=0)

# FINDING THE HIGHEST ACC WITH RANDOM FOREST CLASSIFIER
print('----- Random Forest Classifier -----')
# dictionary to save the best values in certain range
best_acc = {'acc': 0, 'n_estimators': 0}
best_mat = None

# find optimal number for n_estimators in certain range with no max depth
for i in range(1, 50):
    clf = RandomForestClassifier(n_estimators=i, max_depth=None)
    clf.fit(X_train, y_train)
    clf_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, clf_predictions)
    if acc > best_acc['acc']:
        best_acc['acc'] = acc
        best_acc['n_estimators'] = clf.n_estimators
        best_mat = confusion_matrix(y_test, clf_predictions)

print(f'best acc is : {acc} at {best_acc["n_estimators"]} n_estimators')
print(best_mat)

# Neural network
print('----- Neural Network ----- \n (this may take a small moment)')
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# save the best acc and the confusion matrix
nn_best_acc = {'acc': 0, 'neurons_i': 0}
mlp_mat = None

for i in range(15, 36):
    mlp = MLPClassifier(hidden_layer_sizes=i, max_iter=1000)
    mlp.fit(X_train, y_train)

    mlp_predictions = mlp.predict(X_test)
    acc = accuracy_score(y_test, mlp_predictions)
    if acc > nn_best_acc['acc']:
        nn_best_acc['acc'] = acc
        nn_best_acc['neurons_i'] = i
        mlp_mat = confusion_matrix(y_test, mlp_predictions)

print(f'best acc: {nn_best_acc["acc"]} at {nn_best_acc["neurons_i"]} neurons')
print(mlp_mat)

