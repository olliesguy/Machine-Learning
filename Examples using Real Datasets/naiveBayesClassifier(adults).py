import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

# Import the adult.txt file into Python.
data = pd.read_csv('adults.txt', sep=',')

# Convert the string labels to numeric labels.
for label in ['race', 'occupation']:
    data[label] = LabelEncoder().fit_transform(data[label])

# Take the fields of interest and plug them into variable X.
X = data[['race', 'hours_per_week', 'occupation']]
# Make sure to provide the corresponding true values.
Y = data['sex'].values.tolist()

# Split the data into test and training (30% for test).
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Instantiate the classifier.
clf = GaussianNB()

# Train the classifier using the training data.
clf = clf.fit(X_train, y_train)

# Validate the classifier.
accuracy = clf.score(X_test, y_test)
print 'Accuracy: ' + str(accuracy)

# Make a confusion matrix.
prediction = clf.predict(X_test)
cm = confusion_matrix(prediction, y_test)
print cm
