from sklearn import datasets
from sklearn import metrics

# Load the iris datasets
dataset = datasets.load_iris()

# Changes to be made:
# 1. Split dataset into training and test data arrays (currently uses whole dataset)


from sklearn.linear_model import LogisticRegression
# Logistic Regression
logr = LogisticRegression()
logr.fit(dataset.data, dataset.target)
print(logr)

expected = dataset.target
predicted = logr.predict(dataset.data)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

from sklearn.naive_bayes import GaussianNB
# Gaussian-Naive-Bayes
gnb = GaussianNB()
gnb.fit(dataset.data, dataset.target)
print(gnb)

expected = dataset.target
predicted = gnb.predict(dataset.data)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

from sklearn.neighbors import KNeighborsClassifier
# k-Nearest Neighbor
neigh = KNeighborsClassifier()
neigh.fit(dataset.data, dataset.target)
print(neigh)

expected = dataset.target
predicted = neigh.predict(dataset.data)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

from sklearn.tree import DecisionTreeClassifier
# Classification and Regression Trees
cart = DecisionTreeClassifier()
cart.fit(dataset.data, dataset.target)
print(cart)

expected = dataset.target
predicted = cart.predict(dataset.data)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

from sklearn.svm import SVC
# Support Vector Machine
svc = SVC()
svc.fit(dataset.data, dataset.target)
print(model)

# Make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# Summarise the fit of the model.
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
