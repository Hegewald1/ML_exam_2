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
