from sklearn import metrics
from sklearn.svm import SVC

svc = SVC()

# Training dataset for [height, weight, shoe_size].
X_train = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

y_train = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

svc.fit(X_train, y_train)
print(svc)

# Make prediction.
prediction = svc.predict([[190, 70, 43]])

# Summarise the fit of the model.
print(metrics.classification_report(expected, prediction))
print(metrics.confusion_matrix(expected, prediction))
