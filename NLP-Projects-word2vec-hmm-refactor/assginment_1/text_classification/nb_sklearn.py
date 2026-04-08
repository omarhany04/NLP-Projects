from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def sklearn_nb(train_texts, test_texts, y_train):

    vectorizer = CountVectorizer()

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return predictions