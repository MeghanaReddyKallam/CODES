import shap
from pandas import read_csv
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
n_classes=2

n_samples=500
n_informative=2
# print(make_classification)
df = read_csv('/Users/Meghu/Desktop/datafile/4/data4.csv', header=None)
# retrieve the numpy array
data = df.values
# print(data.head())
X, y = data[:, 1:], data[:, 0]
# X, y = make_classification(n_samples=n_samples, n_features=n_features,
#                            n_informative=n_informative, n_redundant=0,
#                            n_classes=n_classes)
# print(X, y)
n_features=14

model = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)

n_test = len(X_test)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

val = accuracy_score(y_test, y_pred)

explainer = shap.TreeExplainer(model)
shap_values = []

for i in range(len(X_test)):
    shap_values.append(explainer.shap_values(X_test[i,:]))
shap_across_all = []

for c in range(n_classes): # Iterate over all classes
    sums = [0] * n_features
    for i in range(n_test): # Iterate over all examples
        for j in range(n_features): # Iterate over all features
            #print(i, j)
            sums[j] += abs(shap_values[i][c][j]) # Take the absolute value
    sums = [x/n_samples for x in sums]
    shap_across_all.append(sums)        

import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
feature_names = [x for x in range(n_features)]
X = np.arange(n_features)
ax.bar(X, shap_across_all[0], color = 'b', width=0.5)
ax.set_ylabel('Average Feature Importance')
ax.set_xlabel('Feature')

plt.show()
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, max_display=10)
