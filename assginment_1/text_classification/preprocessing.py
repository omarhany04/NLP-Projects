from utils.tokenizer import clean_text, tokenize

def preprocess_dataframe(df):

    df['text'] = df['title'] + " " + df['description']

    df['text'] = df['text'].apply(clean_text)

    df['tokens'] = df['text'].apply(tokenize)

    return df