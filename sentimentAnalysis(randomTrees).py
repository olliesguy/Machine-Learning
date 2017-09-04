import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWords2VecUtility import KaggleWords2VecUtility
import pandas as pd
import nltk

if __name__ == '__main__':
    # Step 1. Read the data
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3)
    print "The first review is:"
    print train["review"][0]
    raw_input("Press Enter to continue...")

    # Step 2. Clean the training data
    print 'Download text datasets.'
    nltk.download()
    clean_train_reviews = []
    print "Cleaning and parsing the training set movie review... \n"

    for i in xrange(0, len(train["review"])):
        clean_train_reviews.append(" ".join(
            KaggleWords2VecUtility.review_to_wordlist(train["review"][i], True)
        ))

    # Step 3. Create the bag of words
    print "Creating the bag of words...\n"
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()

    # Step 4. Train the classifier
    print "Training the random forest (this may take a while)...\n"
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train["sentiment"])
    clean_test_reviews = []

    # Step 5. Format the testing data
    print "Cleaning and parsing the test set movie review...\n"
    for i in xrange(0, len(test["review"])):
        clean_test_reviews.append(" ".join(
            KaggleWords2VecUtility.review_to_wordlist(test["review"][i], True)
        ))
    test_data_features = vectorizer.fit_transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Step 6. Predict reviews in the testing data
    print "Predicting test labels...\n"
    result = forest.predict(test_data_features)
    output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)
    print "Wrote results to Bag_of_Words_model.csv"
