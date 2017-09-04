from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=2)

# Training dataset for [height, weight, shoe_size].
X_train = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

y_train = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

rfc = rfc.fit(X_train, y_train)
print(rfc)

prediction = rfc.predict([[190, 70, 43]])

print prediction
