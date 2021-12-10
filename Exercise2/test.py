import pandas as pd
import matplotlib.pyplot as plt

col_names = ["Id", "Churn", "Line", "Grade", "Age", "Distance", "StudyGroup"]
data = pd.read_csv('studentchurn.csv')
# show the data
print(data.describe(include='all'))
# the describe methodd is a great way to get an overview of the data
print(data.values)
print(data.columns)
print(data.shape)
data[['Id', 'Churn', 'Line', 'Grade', 'Age', 'Distance', 'StudyGroup']] = \
    data['Id;Churn;Line;Grade;Age;Distance;StudyGroup'].str.split(';', expand=True)
x = data["Line"]
y = data["Age"]
plt.figure()
plt.scatter(x.values, y.values, color='black', s=20)
plt.show()
x = data["Grade"]
y = data["Age"]
plt.figure()
plt.scatter(x.values, y.values, color='black', s=20)
plt.show()
x = data["Line"]
y = data["Grade"]
plt.figure()
plt.scatter(x.values, y.values, color='black', s=20)
plt.show()
x = data["Grade"]
y = data["Churn"]
plt.figure()
plt.scatter(x.values, y.values, color='black', s=20)
plt.show()

'''
# Finding the best parameters for the Random Forest Classifier
# Start by listing the different parameters we are going to try
n_estimators = [int(x) for x in np.linspace(start=5, stop=50, num=10)]
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

# the random model with Randomized search
clf_random = RandomForestClassifier()
rsCV = RandomizedSearchCV(estimator=clf_random, param_distributions=random_grid, n_iter=30,
                          cv=3, verbose=1, random_state=42, n_jobs=4)  # n_jobs = -1 for all processors
rsCV.fit(X_train, Y_train.values.ravel())
print(rsCV.best_estimator_)
rsCV_predictions = rsCV.best_estimator_.predict(X_test)

print(classification_report(Y_test, rsCV_predictions, target_names=['Completed', 'Stopped']))
importances = pd.DataFrame({'feature': data.columns, 'importance': np.round(rsCV.featureimportances, 3)})
importances = importances.sort_values('importance', ascending=False).set_index('feature')
'''
