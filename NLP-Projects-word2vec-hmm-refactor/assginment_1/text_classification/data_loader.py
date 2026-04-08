from datasets import load_dataset
import pandas as pd

def load_ag_news():
    dataset = load_dataset("sh0416/ag_news")

    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    return train_df, test_df