# -*- coding: utf-8 -*-

from sklearn.datasets import load_digits
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(predictions, test_labels)
    matrix = confusion_matrix(predictions, test_labels)
    print(f'Model Performance - {model.__class__.__name__}')
    print('Accuracy = {:0.4f}%.'.format(accuracy))
    print('Confusion Matrix : \n', matrix)
    return accuracy


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

# ---------- RANDOM FOREST ----------
print('----- Random Forest Classifier -----')

# Finding the best parameters for the Random Forest Classifier
# Start by listing the different parameters we are going to try
n_estimators = [int(x) for x in np.linspace(start=5, stop=250, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# add them to a dictionary
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# the base model in the original file
clf_base = RandomForestClassifier(n_estimators=3, max_depth=None)
clf_base.fit(X_train, y_train)
base_acc = evaluate(clf_base, X_test, y_test)
'''
rf_base_predictions = clf_base.predict(X_test)
misclassifiedIndexes = np.where(y_test!=rf_base_predictions)[0]

fig, ax = plt.subplots(4, 3,figsize=(15,8))
ax = ax.ravel()
for i, badIndex in enumerate(misclassifiedIndexes):
    ax[i].imshow(np.reshape(X_test[badIndex], (8, 8)), cmap=plt.cm.gray)
    ax[i].set_title(f'Predict: {rf_base_predictions[badIndex]}, '
                    f'Actual: {y_test[badIndex]}', fontsize = 10)
    ax[i].set(frame_on=False)
    ax[i].axis('off')
plt.box(False)
plt.axis('off')
'''

# the random model with Randomized search
clf_random = RandomForestClassifier()
rsCV = RandomizedSearchCV(estimator=clf_random, param_distributions=random_grid, n_iter=30,
                          cv=3, verbose=1, random_state=42, n_jobs=4)  # n_jobs = -1 for all processors
rsCV.fit(X_train, y_train)

print('Best params:\n', rsCV.best_params_)
random_acc = evaluate(rsCV.best_estimator_, X_test, y_test)

print('Improvement : base - random = {:0.4f}'.format(random_acc - base_acc))

# ---------- SUPPORT VECTOR MACHINE ----------
random_grid = {'C': [int(x) for x in np.linspace(start=5, stop=250, num=10)],
               'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
               'degree': [int(x) for x in np.linspace(start=3, stop=20, num=10)],
               'gamma': [float(x) for x in np.linspace(start=0.001, stop=0.99, num=10)]}
print(random_grid)
svc = SVC(random_state=1)  # probability=True
svc_random = RandomizedSearchCV(svc, param_distributions=random_grid, n_iter=10, verbose=1,
                                cv=2, n_jobs=4, random_state=42)
svc_random.fit(X_train, y_train)
svc_best_params = svc_random.best_params_
print(svc_best_params)
svc_random_acc = evaluate(svc_random.best_estimator_, X_test, y_test)

# ---------- Neural network ----------
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
