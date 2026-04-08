from data_loader import load_ag_news
from preprocessing import preprocess_dataframe
from nb_from_scratch import NaiveBayesScratch
from nb_sklearn import sklearn_nb

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load Data
train_df, test_df = load_ag_news()

train_df = preprocess_dataframe(train_df)
test_df = preprocess_dataframe(test_df)

# From Scratch NB
nb = NaiveBayesScratch()

nb.fit(train_df['tokens'], train_df['label'])

scratch_preds = nb.predict(test_df['tokens'])

# Sklearn NB
sk_preds = sklearn_nb(
    train_df['text'],
    test_df['text'],
    train_df['label']
)

# Evaluation
def evaluate(name, y_true, y_pred):

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("F1:", f1_score(y_true, y_pred, average='weighted'))


evaluate("Naive Bayes (From Scratch)", test_df['label'], scratch_preds)
evaluate("Naive Bayes (Sklearn)", test_df['label'], sk_preds)