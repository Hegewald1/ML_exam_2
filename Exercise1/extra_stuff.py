from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
import numpy as np
#  https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(predictions, test_labels)
    matrix = confusion_matrix(predictions, test_labels)
    print(f'Model Performance - {model.__class__.__name__}')
    print('Accuracy = {:0.4f}%.'.format(accuracy))
    print('Confusion Matrix : \n', matrix)
    return accuracy


mnist = load_digits()
X_train, X_test, y_train, y_test = train_test_split(mnist.data,
                                                    mnist.target,
                                                    test_size=0.2,
                                                    random_state=0)

# ---------- RANDOM FOREST ----------
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
               'gamma': [float(x) for x in np.linspace(start=0.1, stop=0.99, num=10)]}
print(random_grid)
svc = SVC(random_state=1)  # probability=True
svc_random = RandomizedSearchCV(svc, param_distributions=random_grid, n_iter=10, verbose=1,
                                cv=2, n_jobs=4, random_state=42)
svc_random.fit(X_train, y_train)
svc_best_params = svc_random.best_params_
svc_random_acc = evaluate(svc_random.best_estimator_, X_test, y_test)

# TODO add neural network to this file and update original file
'''
# Decision tree
print('------ Decision Tree -----')

dtc_best_acc = {'acc': 0, 'max_depth': 0}
dtc_mat = None

for i in range(1, 50):
    tree_clf = DecisionTreeClassifier(max_depth=i)
    tree_clf.fit(X_train, y_train)
    dtc_predictions = tree_clf.predict(X_test)
    acc = accuracy_score(y_test, dtc_predictions)
    if acc > dtc_best_acc['acc']:
        dtc_best_acc['acc'] = acc
        dtc_best_acc['max_depth'] = i
        dtc_mat = confusion_matrix(y_test, dtc_predictions)

print(f'best acc: {dtc_best_acc["acc"]} at {dtc_best_acc["max_depth"]} max depth')
print(dtc_mat)
'''
'''
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
'''
