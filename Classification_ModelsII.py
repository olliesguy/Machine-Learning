import unittest

class TestClassifiers(unittest.TestCase):

    # Training dataset for [height, weight, shoe_size].
    X_train = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

    y_train = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

    def test_decision_tree(X_train, y_train):
        from sklearn import tree

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)

        prediction = clf.predict([[190, 70, 43]])
        print 'Decision Tree prediction: ' + prediction

    def test_random_forest(X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=2)
        clf = clf.fit(X_train, y_train)

        prediction = clf.predict([[190, 70, 43]])
        print 'Random Forest prediction: ' + prediction

    def test_k_nearest_neighbour(X_train, y_train):
        from sklearn.neighbors import KNeighborsClassifier

        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X_train, y_train)

        prediction = neigh.predict([190, 70, 43])
        print 'K Nearest Neighbors prediction: ' + prediction

    def test_logisitc_regression(X_train, y_train):
        from sklearn.linear_model import LogisticRegression

        lin_mod = LogisticRegression()
        lin_mod.fit(X_train, y_train)

        prediction = lin_mod.predict([[190, 70, 43]])
        print 'Logistic Regression prediction: ' + prediction

    def test_naive_bayes(X_train, y_train):
        from sklearn.naive_bayes import GaussianNB

        gnb = GaussianNB
        gnb = gnb.fit(X_train, y_train)

        prediction = gnb.predict([[190, 70, 43]])
        print 'Naive-Bayes prediction: ' + prediction

    def test_artificial_neural_network(self):
        # To carry out the following, you need to run cmd ln $ pip3 install scikit-neuralnetwork
        from sknn.mlp import Classifier, Layer
        import numpy as np
        # Training dataset for [height, weight, shoe_size].
        X_train = np.array[[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
             [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

        y_train = np.array['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

        nn = Classifier(
            layers = [
                Layer("Maxout", units=100, pieces=2),
                Layer("Softmax")
            ],
            learning_rate=0.001,
            n_iter=25
        )
        nn.fit(X_train, y_train)

        prediction = nn.predict([[190, 70, 43]])
        print 'Artificial NN: ' + prediction

    def test_ann(self):
        from pybrain.datasets.classification import ClassificationDataSet
        # Line 101 can be replaced with the algorithm of choice,
        # e.g. from pybrain.optimization.hillclimber import HillClimber
        from pybrain.optimization.populationbased.ga import GA
        from pybrain.tools.shortcuts import buildNetwork

        # Create XOR datset.
        d = ClassificationDataSet(2)
        d.addSample([181, 80], [1])
        d.addSample([177, 70], [1])
        d.addSample([160, 60], [0])
        d.addSample([154, 54], [0])
        d.setField('class', [[0.], [1.], [1.], [0.]])

        nn.buildNetwork(2, 3, 1)
        # d.evaluateModuleMSE takes nn as its first and only argument.
        ga = GA(d.evaluateModuleMSE, nn, minimize=True)
        for i in range(100):
            nn = ga.learn(0)[0]

        print nn.active([181, 80])
