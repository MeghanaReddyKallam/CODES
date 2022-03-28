import shap
import xgboost
import shap
from pandas import read_csv
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
n_classes=2
n_features=10
n_samples=300
n_informative=2
# print(make_classification)
df = read_csv('/Users/praneethakishorekumar/Desktop/11/features11.csv', header=None)
# retrieve the numpy array
data = df.values
# print(data.head())
X, y = data[:, 1:], data[:, 0]
# X, y = make_classification(n_samples=n_samples, n_features=n_features,
#                            n_informative=n_informative, n_redundant=0,
#                            n_classes=n_classes)
# print(X, y)






# train XGBoost model
X, y = data[:, 1:], data[:, 0]
model = xgboost.XGBClassifier().fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)
n_test = len(X_test)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
print(y_pred, y_prob)
accuracy_score(y_test, y_pred)

explainer = shap.Explainer(model, X)

# compute SHAP values

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




import shap
import xgboost
import shap
from pandas import read_csv
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
n_classes=2
n_features=10
n_samples=300
n_informative=2
# print(make_classification)
df = read_csv('//Users/Meghu/Desktop/datafile/11/data11.csv', header=None)
# retrieve the numpy array
data = df.values
# print(data.head())
X, y = data[:, 1:], data[:, 0]
# X, y = make_classification(n_samples=n_samples, n_features=n_features,
#                            n_informative=n_informative, n_redundant=0,
#                            n_classes=n_classes)
# print(X, y)






# train XGBoost model
X, y = data[:, 1:], data[:, 0]
model = xgboost.XGBClassifier().fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)
n_test = len(X_test)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
print(y_pred, y_prob)
accuracy_score(y_test, y_pred)

explainer = shap.Explainer(model, X)

# compute SHAP values

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


shap.plots.bar(shap_values)

